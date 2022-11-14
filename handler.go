package main

import (
	"github.com/qinsheng99/go-py-message/app"
	"github.com/qinsheng99/go-py-message/infrastructure/message"
	"github.com/sirupsen/logrus"
)

type handler struct {
	log       *logrus.Entry
	evaluate  app.EvaluateService
	calculate app.CalculateService
}

func (h *handler) Calculate(cal message.Calculate, res *message.ScoreRes) error {
	return h.calculate.Calculate(cal, res)
}

func (h *handler) Evaluate(cal message.Evaluate, res *message.ScoreRes) error {
	return h.evaluate.Evaluate(cal, res)
}
