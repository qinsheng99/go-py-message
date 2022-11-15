package test

import (
	"encoding/json"
	"testing"

	"github.com/opensourceways/community-robot-lib/kafka"
	"github.com/opensourceways/community-robot-lib/mq"
	"github.com/qinsheng99/go-py-message/infrastructure/message"
)

func TestMqGame(t *testing.T) {
	err := kafka.Init()
	if err != nil {
		t.Fatal(err)
	}

	err = kafka.Connect()
	if err != nil {
		t.Fatal(err)
	}

	data1 := message.Game{
		GameType: message.GameType{Type: "text", UserId: 1},
		GameFields: message.GameFields{
			PredPath: "xihe-obj/competitions/昇思AI挑战赛-多类别图像分类/submit_result/s9qfqri3zpc8j2x7_1/result_example_5120-2022-8-8-15-3-16.txt",
			//TruePath: "xihe-obj/competitions/昇思AI挑战赛-多类别图像分类/result/label.txt",
			Cls: 256,
			Pos: 1,
		},
	}

	data2 := message.Game{
		GameType: message.GameType{Type: "image", UserId: 2},
		GameFields: message.GameFields{
			PredPath: "xihe-obj/competitions/昇思AI挑战赛-多类别图像分类/submit_result/s9qfqri3zpc8j2x7_1/result_example_5120-2022-8-8-15-3-16.txt",
			//TruePath: "xihe-obj/competitions/昇思AI挑战赛-多类别图像分类/result/label.txt",
			Cls: 256,
			Pos: 1,
		},
	}

	data3 := message.Game{
		GameType: message.GameType{Type: "style", UserId: 3},
		GameFields: message.GameFields{
			UserResult: "xihe-obj/competitions/昇思AI挑战赛-艺术家画作风格迁移/submit_result/victor_1/result",
		},
	}

	data4 := message.Game{
		GameType: message.GameType{Type: "dd", UserId: 4},
		GameFields: message.GameFields{
			UserResult: "xihe-obj/competitions/昇思AI挑战赛-艺术家画作风格迁移/submit_result/victor_1/result",
		},
	}

	bys1, err := json.Marshal(data1)
	if err != nil {
		t.Fatal(err)
	}
	bys2, err := json.Marshal(data2)
	if err != nil {
		t.Fatal(err)
	}
	bys3, err := json.Marshal(data3)
	if err != nil {
		t.Fatal(err)
	}

	bys4, err := json.Marshal(data4)
	if err != nil {
		t.Fatal(err)
	}

	msg1 := mq.Message{Body: bys1}
	msg2 := mq.Message{Body: bys2}
	msg3 := mq.Message{Body: bys3}
	msg4 := mq.Message{Body: bys4}

	err = kafka.Publish("game", &msg1)
	if err != nil {
		t.Fatal(err)
	}

	err = kafka.Publish("game", &msg3)
	if err != nil {
		t.Fatal(err)
	}

	err = kafka.Publish("game", &msg2)
	if err != nil {
		t.Fatal(err)
	}

	err = kafka.Publish("game", &msg4)
	if err != nil {
		t.Fatal(err)
	}

	err = kafka.Disconnect()
	if err != nil {
		t.Fatal(err)
	}
}
