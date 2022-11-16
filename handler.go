package main

import (
	"time"

	"github.com/qinsheng99/go-py-message/app"
	"github.com/qinsheng99/go-py-message/config"
	"github.com/qinsheng99/go-py-message/infrastructure/message"
	"github.com/sirupsen/logrus"
)

type handler struct {
	log       *logrus.Entry
	maxRetry  int
	evaluate  app.EvaluateService
	calculate app.CalculateService
	match     config.MatchImpl
}

type handlerMessage struct {
	message.MatchMessage
	score float64
	msg   string
}

const sleepTime = 100 * time.Millisecond

func (h *handler) Calculate(cal *message.MatchMessage, match *message.MatchFields) error {
	return h.do(func(b bool) error {
		var res message.ScoreRes
		var m = handlerMessage{MatchMessage: *cal}
		err := h.calculate.Calculate(match, &res)
		if err != nil {
			h.log.Errorf("calculate script failed,err: %v", err)
			m.msg = err.Error()
		} else {
			if res.Status != 200 {
				m.msg = res.Msg
			} else {
				m.score = res.Data
			}
		}
		h.handlerCalculate(m)
		return err
	})
}

func (h *handler) Evaluate(eval *message.MatchMessage, match *message.MatchFields) error {
	return h.do(func(b bool) error {
		var res message.ScoreRes
		var m = handlerMessage{MatchMessage: *eval}
		err := h.evaluate.Evaluate(match, &res)
		if err != nil {
			h.log.Errorf("evaluate script failed,err: %v", err)
			m.msg = err.Error()
		} else {
			if res.Status != 200 {
				m.msg = res.Msg
			} else {
				m.score = res.Metrics.Acc
			}
		}
		h.handlerEvaluate(m)
		return err
	})
}

func (h *handler) GetMatch(id int) *config.Match {
	return h.match.GetMatch(id)
}

func (h *handler) handlerCalculate(m handlerMessage) {
	h.log.Infof("call calculate rpc game type:%d,user:%v,stage:%v,res:(%s/%v)", m.MatchId, m.UserId, m.MatchStage, m.msg, m.score)
}

func (h *handler) handlerEvaluate(m handlerMessage) {
	h.log.Infof("call evaluate rpc game type:%d,user:%v,stage:%v,res:(%s/%v)", m.MatchId, m.UserId, m.MatchStage, m.msg, m.score)
}

func (h *handler) do(f func(bool) error) (err error) {
	return h.retry(f, sleepTime)
}

func (h *handler) retry(f func(bool) error, interval time.Duration) (err error) {
	n := h.maxRetry - 1

	if err = f(n <= 0); err == nil || n <= 0 {
		return
	}

	for i := 1; i < n; i++ {
		time.Sleep(interval)

		if err = f(false); err == nil {
			return
		}
	}

	time.Sleep(interval)

	return f(true)
}
