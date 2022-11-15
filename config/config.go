package config

import (
	"regexp"
	"strings"

	"github.com/opensourceways/community-robot-lib/mq"
	"github.com/opensourceways/community-robot-lib/utils"
)

var reIpPort = regexp.MustCompile(`^((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.?\b){4}:[1-9][0-9]*$`)

type Configuration struct {
	MaxRetry        int    `json:"max_retry"   required:"true"`
	ImageAnswerPath string `json:"image_answer_path" required:"true"`
	TextAnswerPath  string `json:"text_answer_path" required:"true"`
	MQ              MQ     `json:"mq"          required:"true"`
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
	Game string `json:"game"         required:"true"`
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
