package score

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"

	"github.com/qinsheng99/go-py-message/domain/score"
	"github.com/qinsheng99/go-py-message/infrastructure/message"
)

type calculateImpl struct {
	calculate string
}

func NewCalculateScore(calculate string) score.CalculateScore {
	return &calculateImpl{
		calculate: calculate,
	}
}

type evaluateImpl struct {
	evaluate string
}

func NewEvaluateScore(evaluate string) score.EvaluateScore {
	return &evaluateImpl{
		evaluate: evaluate,
	}
}

func (s *evaluateImpl) Evaluate(col message.Evaluate) (data []byte, err error) {
	args := []string{s.evaluate, "--pred_path", col.PredPath, "--true_path", col.TruePath, "--cls", strconv.Itoa(col.Cls), "--pos", strconv.Itoa(col.Pos)}
	data, err = exec.Command("python3", args...).Output()

	if err != nil {
		return
	}
	data = bytes.ReplaceAll(bytes.TrimSpace(data), []byte(`'`), []byte(`"`))
	return
}

func (s *calculateImpl) Calculate(col message.Calculate) (data []byte, err error) {
	path := filepath.Join(os.Getenv("UPLOAD"), col.UserName)
	defer os.RemoveAll(path)
	args := []string{s.calculate, "--user_result", col.UserResult, "--unzip_path", path}
	data, err = exec.Command("python3", args...).Output()

	if err != nil {
		return
	}
	data = bytes.ReplaceAll(bytes.TrimSpace(data), []byte(`'`), []byte(`"`))
	return
}
