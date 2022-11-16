package main

import (
	"time"

	"github.com/qinsheng99/go-py-message/app"
	"github.com/qinsheng99/go-py-message/infrastructure/message"
	"github.com/sirupsen/logrus"
)

type handler struct {
	log             *logrus.Entry
	maxRetry        int
	evaluate        app.EvaluateService
	calculate       app.CalculateService
	textAnswerPath  string
	imageAnswerPath string
}

type handlerMessage struct {
	message.GameType
	score float64
	msg   string
}

const sleepTime = 100 * time.Millisecond

func (h *handler) Calculate(cal *message.Game) error {
	return h.do(func(b bool) error {
		var res message.ScoreRes
		err := h.calculate.Calculate(&cal.GameFields, &res)
		var m handlerMessage
		m.GameType = cal.GameType
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

func (h *handler) Evaluate(eval *message.Game) error {
	var path string
	switch eval.Type {
	case message.Image:
		path = h.imageAnswerPath
	case message.Text:
		path = h.textAnswerPath

	}
	return h.do(func(b bool) error {
		var res message.ScoreRes
		err := h.evaluate.Evaluate(&eval.GameFields, &res, path)
		var m handlerMessage
		m.GameType = eval.GameType
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

func (h *handler) handlerCalculate(m handlerMessage) {
	h.log.Infof("call calculate rpc game type:%s,user:%v,res:(%s/%v)", m.Type, m.UserId, m.msg, m.score)
}

func (h *handler) handlerEvaluate(m handlerMessage) {
	h.log.Infof("call calculate rpc game type:%s,user:%v,res:(%s/%v)", m.Type, m.UserId, m.msg, m.score)
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
