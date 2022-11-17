package message

// GameFields Path 用户上传的result.txt
// Cls 比赛的类别数
// Pos 类别索引标签的起始位
// UserResult  存有1000张图片的zip文件

// MatchMessage
// 文本分类 text  图像分类 image  风格迁移style
type MatchMessage struct {
	CompetitionId string `json:"competition_id"`
	UserId        string `json:"submission_id"`
	Phase         string `json:"phase"`
	Path          string `json:"path,omitempty"`
}

type MatchFields struct {
	Path           string `json:"path"`
	AnswerPath     string `json:"answer_path"`
	FidWeightsPath string `json:"fid_weights_path"`
	RealPath       string `json:"real_path"`
	Cls, Pos       int
}

type ScoreRes struct {
	Status  int     `json:"status"`
	Msg     string  `json:"msg"`
	Data    float32 `json:"data,om64itempty"`
	Metrics metrics `json:"metrics,omitempty"`
}
type metrics struct {
	Ap   float32 `json:"ap,omitempty"`
	Ar   float32 `json:"ar,omitempty"`
	Af1  float32 `json:"af1,omitempty"`
	Af05 float32 `json:"af05,omitempty"`
	Af2  float32 `json:"af2,omitempty"`
	Acc  float32 `json:"acc,omitempty"`
	Err  float32 `json:"err,omitempty"`
}

type MatchImpl interface {
	Calculate(*MatchMessage, *MatchFields) error
	Evaluate(*MatchMessage, *MatchFields) error
	GetMatch(id string) MatchFieldImpl
}

type MatchFieldImpl interface {
	GetAnswerPath() string
	GetType() string
	GetPos() int
	GetCls() int
	GetFidWeightsPath() string
	GetRealPath() string
}
