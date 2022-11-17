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
	Id             string `json:"competition_id" required:"true"`
	Type           string `json:"competition_type" required:"true"`
	AnswerPath     string `json:"answer_path"`
	FidWeightsPath string `json:"fid_weights_path"`
	RealPath       string `json:"real_path"`
	Pos            int    `json:"pos"`
	Cls            int    `json:"cls"`
}

func (m *match) GetAnswerPath() string {
	return m.AnswerPath
}

func (m *match) GetFidWeightsPath() string {
	return m.FidWeightsPath
}

func (m *match) GetRealPath() string {
	return m.RealPath
}

func (m *match) GetType() string {
	return m.Type
}

func (m *match) GetPos() int {
	return m.Pos
}

func (m *match) GetCls() int {
	return m.Cls
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
	GetAnswerPath() string
	GetType() string
	GetPos() int
	GetCls() int
	GetFidWeightsPath() string
	GetRealPath() string
}
