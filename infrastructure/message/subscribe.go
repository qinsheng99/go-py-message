package message

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/opensourceways/community-robot-lib/kafka"
	"github.com/opensourceways/community-robot-lib/mq"
	"github.com/sirupsen/logrus"
)

func Subscribe(ctx context.Context, handler interface{}, log *logrus.Entry) error {
	subscribers := make(map[string]mq.Subscriber)

	defer func() {
		for k, s := range subscribers {
			if err := s.Unsubscribe(); err != nil {
				log.Errorf("failed to unsubscribe for topic:%s, err:%v", k, err)
			}
		}
	}()

	s, err := registerHandlerForEvaluate(handler)
	if err != nil {
		return err
	}
	if s != nil {
		subscribers[s.Topic()] = s
	}

	s, err = registerHandlerForCalculate(handler)
	if err != nil {
		return err
	}
	if s != nil {
		subscribers[s.Topic()] = s
	}

	// register end
	if len(subscribers) == 0 {
		return nil
	}

	<-ctx.Done()

	return nil
}

func registerHandlerForEvaluate(handler interface{}) (mq.Subscriber, error) {
	h, ok := handler.(EvaluateImpl)
	if !ok {
		return nil, nil
	}

	return kafka.Subscribe(topics.Evaluate, func(e mq.Event) (err error) {
		msg := e.Message()
		if msg == nil {
			return
		}

		body := Evaluate{}
		if err = json.Unmarshal(msg.Body, &body); err != nil {
			return
		}

		var res ScoreRes

		err = h.Evaluate(body, &res)
		if err != nil {
			return err
		}

		fmt.Println(res)

		return nil
	})
}

func registerHandlerForCalculate(handler interface{}) (mq.Subscriber, error) {
	h, ok := handler.(CalculateImpl)
	if !ok {
		return nil, nil
	}

	return kafka.Subscribe(topics.Calculate, func(e mq.Event) (err error) {
		msg := e.Message()
		if msg == nil {
			return
		}

		body := Calculate{}
		if err = json.Unmarshal(msg.Body, &body); err != nil {
			return
		}

		var res ScoreRes

		err = h.Calculate(body, &res)
		if err != nil {
			return err
		}

		fmt.Println(res)

		return nil
	})
}
