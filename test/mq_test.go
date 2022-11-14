package test

import (
	"encoding/json"
	"testing"

	"github.com/opensourceways/community-robot-lib/kafka"
	"github.com/opensourceways/community-robot-lib/mq"
	"github.com/qinsheng99/go-py-message/infrastructure/message"
)

func TestMqCal(t *testing.T) {
	err := kafka.Init()
	if err != nil {
		t.Fatal(err)
	}

	err = kafka.Connect()
	if err != nil {
		t.Fatal(err)
	}

	data := message.Calculate{
		UserResult: "xihe-obj/competitions/昇思AI挑战赛-艺术家画作风格迁移/submit_result/victor_1/result",
		UserName:   "ceshi",
	}

	bys, err := json.Marshal(data)
	if err != nil {
		t.Fatal(err)
	}

	msg := mq.Message{Body: bys}

	err = kafka.Publish("calculate", &msg)
	if err != nil {
		t.Fatal(err)
	}

	err = kafka.Disconnect()
	if err != nil {
		t.Fatal(err)
	}
}

func TestMqEval(t *testing.T) {
	err := kafka.Init()
	if err != nil {
		t.Fatal(err)
	}

	err = kafka.Connect()
	if err != nil {
		t.Fatal(err)
	}

	data := message.Evaluate{
		PredPath: "xihe-obj/competitions/昇思AI挑战赛-多类别图像分类/submit_result/s9qfqri3zpc8j2x7_1/result_example_5120-2022-8-8-15-3-16.txt",
		TruePath: "xihe-obj/competitions/昇思AI挑战赛-多类别图像分类/result/label.txt",
		Cls:      256,
		Pos:      1,
		UserName: "ceshi",
	}

	bys, err := json.Marshal(data)
	if err != nil {
		t.Fatal(err)
	}

	msg := mq.Message{Body: bys}

	err = kafka.Publish("evaluate", &msg)
	if err != nil {
		t.Fatal(err)
	}

	err = kafka.Disconnect()
	if err != nil {
		t.Fatal(err)
	}
}
