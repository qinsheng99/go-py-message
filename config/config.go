package config

import (
	"regexp"
	"strings"

	"github.com/opensourceways/community-robot-lib/mq"
	"github.com/opensourceways/community-robot-lib/utils"
)

var reIpPort = regexp.MustCompile(`^((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.?\b){4}:[1-9]\d*$`)

type Configuration struct {
	MaxRetry int     `json:"max_retry"         required:"true"`
	Matchs   []match `json:"matchs"             required:"true"`
	Endpoint string  `json:"endpoint"          required:"true"`
	MQ       MQ      `json:"mq"                required:"true"`
}

type match struct {
	Id                        string `json:"competition_id" required:"true"`
	AnswerFinalPath           string `json:"answer_final_path"`
	AnswerPreliminaryPath     string `json:"answer_preliminary_path"`
	FidWeightsFinalPath       string `json:"fid_weights_final_path"`
	FidWeightsPreliminaryPath string `json:"fid_weights_preliminary_path"`
	RealFinalPath             string `json:"real_final_path"`
	RealPreliminaryPath       string `json:"real_preliminary_path"`
	Pos                       int    `json:"pos"`
	Cls                       int    `json:"cls"`
	Prefix                    string `json:"prefix" required:"true"`
}

func (m *match) GetAnswerFinalPath() string {
	return m.AnswerFinalPath
}

func (m *match) GetAnswerPreliminaryPath() string {
	return m.AnswerPreliminaryPath
}

func (m *match) GetPrefix() string {
	return m.Prefix
}

func (m *match) GetFidWeightsFinalPath() string {
	return m.FidWeightsFinalPath
}

func (m *match) GetFidWeightsPreliminaryPath() string {
	return m.FidWeightsPreliminaryPath
}

func (m *match) GetRealFinalPath() string {
	return m.RealFinalPath
}

func (m *match) GetRealPreliminaryPath() string {
	return m.RealPreliminaryPath
}

func (m *match) GetPos() int {
	return m.Pos
}

func (m *match) GetCls() int {
	return m.Cls
}

func (m *match) GetCompetitionId() string {
	return m.Id
}

func (cfg *Configuration) GetMatch(id string) MatchFieldImpl {
	for k := range cfg.Matchs {
		m := &cfg.Matchs[k]
		if strings.EqualFold(m.Id, id) {
			return m
		}
	}
	return nil
}

func (cfg *Configuration) GetMQConfig() mq.MQConfig {
	return mq.MQConfig{
		Addresses: cfg.MQ.ParseAddress(),
	}
}

func (cfg *Configuration) Validate() error {
	if _, err := utils.BuildRequestBody(cfg, ""); err != nil {
		return err
	}

	return nil
}

func (cfg *Configuration) SetDefault() {
	if cfg.MaxRetry <= 0 {
		cfg.MaxRetry = 10
	}
}

func (cfg *MQ) ParseAddress() []string {
	v := strings.Split(cfg.Address, ",")
	r := make([]string, 0, len(v))
	for i := range v {
		if reIpPort.MatchString(v[i]) {
			r = append(r, v[i])
		}
	}

	return r
}

type MQ struct {
	Address string `json:"address" required:"true"`
	Topics  Topics `json:"topics"  required:"true"`
}

type Topics struct {
	Match string `json:"submission"       required:"true"`
}

func LoadConfig(path string, cfg interface{}) error {
	if err := utils.LoadFromYaml(path, cfg); err != nil {
		return err
	}

	if f, ok := cfg.(SetDefault); ok {
		f.SetDefault()
	}

	if f, ok := cfg.(Validate); ok {
		if err := f.Validate(); err != nil {
			return err
		}
	}

	return nil
}

type Validate interface {
	Validate() error
}

type SetDefault interface {
	SetDefault()
}

type MatchImpl interface {
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
