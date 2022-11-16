package app

import (
	"bytes"
	"encoding/json"

	"github.com/qinsheng99/go-py-message/domain/score"
	"github.com/qinsheng99/go-py-message/infrastructure/message"
)

type scoreService struct {
	c score.CalculateScore
	e score.EvaluateScore
}

type CalculateService interface {
	Calculate(*message.MatchFields, *message.ScoreRes) error
}

func NewCalculateService(c score.CalculateScore) CalculateService {
	return &scoreService{
		c: c,
	}
}

type EvaluateService interface {
	Evaluate(*message.MatchFields, *message.ScoreRes) error
}

func NewEvaluateService(e score.EvaluateScore) EvaluateService {
	return &scoreService{
		e: e,
	}
}

func (s *scoreService) Evaluate(col *message.MatchFields, res *message.ScoreRes) error {
	bys, err := s.e.Evaluate(col)
	if err != nil {
		return err
	}

	return json.NewDecoder(bytes.NewBuffer(bys)).Decode(res)
}

func (s *scoreService) Calculate(col *message.MatchFields, res *message.ScoreRes) error {
	bys, err := s.c.Calculate(col)
	if err != nil {
		return err
	}

	return json.NewDecoder(bytes.NewBuffer(bys)).Decode(res)
}
