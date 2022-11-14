package score

import (
	"github.com/qinsheng99/go-py-message/infrastructure/message"
)

type CalculateScore interface {
	Calculate(message.Calculate) ([]byte, error)
}

type EvaluateScore interface {
	Evaluate(message.Evaluate) ([]byte, error)
}
