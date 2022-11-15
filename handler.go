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

const sleepTime = 100 * time.Millisecond

func (h *handler) Calculate(cal *message.GameFields, res *message.ScoreRes) error {
	return h.do(func(b bool) error {
		err := h.calculate.Calculate(cal, res)
		if err != nil {
			h.log.Error(err)
			return err
		}

		return nil
	})
}

func (h *handler) Evaluate(cal *message.GameFields, res *message.ScoreRes, typ string) error {
	var path string
	switch typ {
	case message.Image:
		path = h.imageAnswerPath
	case message.Text:
		path = h.textAnswerPath

	}
	return h.do(func(b bool) error {
		err := h.evaluate.Evaluate(cal, res, path)
		if err != nil {
			h.log.Error(err)
			return err
		}

		return nil
	})
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
