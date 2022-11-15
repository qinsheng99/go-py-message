package score

import (
	"github.com/qinsheng99/go-py-message/infrastructure/message"
)

type CalculateScore interface {
	Calculate(*message.GameFields) ([]byte, error)
}

type EvaluateScore interface {
	Evaluate(*message.GameFields, string) ([]byte, error)
}
