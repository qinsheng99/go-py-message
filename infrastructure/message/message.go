package message

// Evaluate PredPath 用户上传的result.txt
// TruePath 标准答案的result.txt
// cls 比赛的类别数
// pos 类别索引标签的起始位
type Evaluate struct {
	PredPath string `json:"y_pred_path"`
	TruePath string `json:"y_true_path"`
	UserName string `json:"user_name"`
	Cls, Pos int
}

// Calculate UserResult 存有1000张图片的zip文件
// UserName 用户名
type Calculate struct {
	UserResult string `json:"user_result"`
	UserName   string `json:"user_name"`
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

type CalculateImpl interface {
	Calculate(Calculate, *ScoreRes) error
}

type EvaluateImpl interface {
	Evaluate(Evaluate, *ScoreRes) error
}
