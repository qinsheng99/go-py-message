package message

type MatchMessage struct {
	CompetitionId string `json:"cid"`
	UserId        string `json:"id"`
	Phase         string `json:"phase"`
	Path          string `json:"obs_path"`
	PlayerId      string `json:"pid"`
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
	Data    float32 `json:"data"`
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
	GetAnswerFinalPath() string
	GetAnswerPreliminaryPath() string
	GetPos() int
	GetCls() int
	GetFidWeightsFinalPath() string
	GetFidWeightsPreliminaryPath() string
	GetRealFinalPath() string
	GetRealPreliminaryPath() string
	GetPrefix() string
	GetCompetitionId() string
}
