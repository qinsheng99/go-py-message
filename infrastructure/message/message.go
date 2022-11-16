package message

import (
	"github.com/qinsheng99/go-py-message/config"
)

// GameFields Path 用户上传的result.txt
// Cls 比赛的类别数
// Pos 类别索引标签的起始位
// UserResult  存有1000张图片的zip文件

// MatchMessage
// 文本分类 text  图像分类 image  风格迁移style
type MatchMessage struct {
	MatchId    int    `json:"match_id"`
	UserId     int    `json:"user_id"`
	MatchStage int    `json:"match_stage"`
	Path       string `json:"path"`
}

type MatchFields struct {
	Path       string `json:"path"`
	AnswerPath string `json:"answer_path"`
	Cls, Pos   int
}

type ScoreRes struct {
	Status  int     `json:"status"`
	Msg     string  `json:"msg"`
	Data    float64 `json:"data,omitempty"`
	Metrics metrics `json:"metrics,omitempty"`
}
type metrics struct {
	Ap   float64 `json:"ap,omitempty"`
	Ar   float64 `json:"ar,omitempty"`
	Af1  float64 `json:"af1,omitempty"`
	Af05 float64 `json:"af05,omitempty"`
	Af2  float64 `json:"af2,omitempty"`
	Acc  float64 `json:"acc,omitempty"`
	Err  float64 `json:"err,omitempty"`
}

type MatchImpl interface {
	Calculate(*MatchMessage, *MatchFields) error
	Evaluate(*MatchMessage, *MatchFields) error
	GetMatch(id int) *config.Match
}

type MatchFieldImpl interface {
	GetAnswerPath() string
	GetType() string
	GetPos() int
	GetCls() int
}
