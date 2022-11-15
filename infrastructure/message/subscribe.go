package message

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/opensourceways/community-robot-lib/kafka"
	"github.com/opensourceways/community-robot-lib/mq"
	"github.com/sirupsen/logrus"
)

const (
	Image = "image"
	Text  = "text"
	Style = "style"
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

	s, err := registerHandlerForGame(handler)
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

func registerHandlerForGame(handler interface{}) (mq.Subscriber, error) {
	h, ok := handler.(GameImpl)
	if !ok {
		return nil, nil
	}

	return kafka.Subscribe(topics.Game, func(e mq.Event) (err error) {
		msg := e.Message()
		if msg == nil {
			return
		}

		body := Game{}
		if err = json.Unmarshal(msg.Body, &body); err != nil {
			return
		}

		switch body.Type {
		case Text, Image:
			go evaluate(h, &body)
		case Style:
			go calculate(h, &body)
		default:
			return fmt.Errorf("unknown type: %s", body.Type)
		}

		return nil
	})
}

func evaluate(h GameImpl, body *Game) {
	var res ScoreRes
	err := h.Evaluate(&body.GameFields, &res)
	if err != nil {
		logrus.Errorf("evaluate failed, game type:%s,user:%v", body.Type, body.UserId)
	}

	logrus.Infof("game type:%s,user:%v,res:%v", body.Type, body.UserId, res)
}

func calculate(h GameImpl, body *Game) {
	var res ScoreRes
	err := h.Calculate(&body.GameFields, &res)
	if err != nil {
		logrus.Errorf("evaluate failed, game type:%s,user:%v", body.Type, body.UserId)
	}

	logrus.Infof("game type:%s,user:%v,res:%v", body.Type, body.UserId, res)
}
