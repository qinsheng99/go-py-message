package message

// GameFields PredPath 用户上传的result.txt
// Cls 比赛的类别数
// Pos 类别索引标签的起始位
// UserResult  存有1000张图片的zip文件
type GameFields struct {
	UserResult string `json:"user_result"`
	PredPath   string `json:"pred_path"`
	Cls, Pos   int
}

// GameType
// 文本分类 text  图像分类 image  风格迁移style
type GameType struct {
	Type      string `json:"type"`
	UserId    int    `json:"user_id"`
	GameState int    `json:"game_state"`
}

type Game struct {
	GameType
	GameFields
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

type GameImpl interface {
	Calculate(*Game, *ScoreRes) error
	Evaluate(*Game, *ScoreRes, string) error
}
