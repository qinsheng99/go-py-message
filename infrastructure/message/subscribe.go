package message

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/opensourceways/community-robot-lib/kafka"
	"github.com/opensourceways/community-robot-lib/mq"
	"github.com/sirupsen/logrus"
)

const (
	StyleCompetition = "1"
	TextCompetition  = "2"
	ImageCompetition = "3"
	LearnText        = "learn23-text"
	LeanImg          = "learn23-img"

	CompetitionPhaseFinal       = "final"
	CompetitionPhasePreliminary = "preliminary"
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

		switch m.GetCompetitionId() {
		case TextCompetition, ImageCompetition, LearnText, LeanImg:
			go evaluate(h, &body, m)
		case StyleCompetition:
			go calculate(h, &body, m)
		}

		return nil
	})
}

func evaluate(h MatchImpl, body *MatchMessage, m MatchFieldImpl) {
	var c = MatchFields{Path: m.GetPrefix() + "/" + strings.TrimPrefix(body.Path, "/"), Cls: m.GetCls(), Pos: m.GetPos()}
	switch body.Phase {
	case CompetitionPhaseFinal:
		c.AnswerPath = m.GetAnswerFinalPath()
	case CompetitionPhasePreliminary:
		c.AnswerPath = m.GetAnswerPreliminaryPath()
	}
	logrus.Info(c)
	err := h.Evaluate(body, &c)
	if err != nil {
		logrus.Errorf("evaluate failed, competition id:%s,user:%v", body.CompetitionId, body.UserId)
	}
}

func calculate(h MatchImpl, body *MatchMessage, m MatchFieldImpl) {
	var c = MatchFields{Path: m.GetPrefix() + "/" + strings.TrimPrefix(body.Path, "/")}
	switch body.Phase {
	case CompetitionPhaseFinal:
		c.FidWeightsPath = m.GetFidWeightsFinalPath()
		c.RealPath = m.GetRealFinalPath()
	case CompetitionPhasePreliminary:
		c.FidWeightsPath = m.GetFidWeightsPreliminaryPath()
		c.RealPath = m.GetRealPreliminaryPath()
	}
	logrus.Info(c)
	err := h.Calculate(body, &c)
	if err != nil {
		logrus.Errorf("evaluate failed, competition id:%s,user:%v", body.CompetitionId, body.UserId)
	}
}
