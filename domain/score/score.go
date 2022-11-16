package score

import (
	"github.com/qinsheng99/go-py-message/infrastructure/message"
)

type CalculateScore interface {
	Calculate(*message.MatchFields) ([]byte, error)
}

type EvaluateScore interface {
	Evaluate(*message.MatchFields) ([]byte, error)
}
