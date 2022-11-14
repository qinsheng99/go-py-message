package message

import (
	"github.com/opensourceways/community-robot-lib/kafka"
	"github.com/opensourceways/community-robot-lib/mq"
	"github.com/qinsheng99/go-py-message/config"
	"github.com/sirupsen/logrus"
)

var topics config.Topics

func Init(cfg mq.MQConfig, log *logrus.Entry, topic config.Topics) error {
	topics = topic
	err := kafka.Init(
		mq.Addresses(cfg.Addresses...),
		mq.Log(log),
	)
	if err != nil {
		return err
	}

	return kafka.Connect()
}

func Exit(log *logrus.Entry) {
	if err := kafka.Disconnect(); err != nil {
		log.Errorf("exit kafka, err:%v", err)
	}
}
