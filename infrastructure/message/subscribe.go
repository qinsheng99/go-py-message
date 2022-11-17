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
	log.Info("listen mq")

	<-ctx.Done()

	return nil
}

func registerHandlerForGame(handler interface{}) (mq.Subscriber, error) {
	h, ok := handler.(MatchImpl)
	if !ok {
		return nil, nil
	}

	return kafka.Subscribe(topics.Match, func(e mq.Event) (err error) {
		msg := e.Message()
		if msg == nil {
			return
		}

		body := MatchMessage{}
		if err = json.Unmarshal(msg.Body, &body); err != nil {
			return
		}

		m := h.GetMatch(body.CompetitionId)
		if m == nil {
			return fmt.Errorf("unknown competition id:%s", body.CompetitionId)
		}

		switch m.GetType() {
		case Text, Image:
			go evaluate(h, &body, m)
		case Style:
			go calculate(h, &body)
		}

		return nil
	})
}

func evaluate(h MatchImpl, body *MatchMessage, m MatchFieldImpl) {
	var c = MatchFields{Path: body.Path, Cls: m.GetCls(), Pos: m.GetPos(), AnswerPath: m.GetAnswerPath()}
	err := h.Evaluate(body, &c)
	if err != nil {
		logrus.Errorf("evaluate failed, competition id:%s,user:%v", body.CompetitionId, body.UserId)
	}
}

func calculate(h MatchImpl, body *MatchMessage) {
	var c = MatchFields{Path: body.Path}
	err := h.Calculate(body, &c)
	if err != nil {
		logrus.Errorf("evaluate failed, competition id:%s,user:%v", body.CompetitionId, body.UserId)
	}
}
