package main

import (
	"context"
	"flag"
	"os"
	"os/signal"
	"sync"
	"syscall"

	"github.com/opensourceways/community-robot-lib/logrusutil"
	liboptions "github.com/opensourceways/community-robot-lib/options"
	"github.com/opensourceways/xihe-grpc-protocol/grpc/client"
	"github.com/qinsheng99/go-py-message/app"
	"github.com/qinsheng99/go-py-message/config"
	"github.com/qinsheng99/go-py-message/infrastructure/message"
	"github.com/qinsheng99/go-py-message/infrastructure/score"
	"github.com/sirupsen/logrus"
)

type options struct {
	service     liboptions.ServiceOptions
	enableDebug bool
}

func (o *options) Validate() error {
	return o.service.Validate()
}

func gatherOptions(fs *flag.FlagSet, args ...string) options {
	var o options

	o.service.AddFlags(fs)

	fs.BoolVar(
		&o.enableDebug, "enable_debug", false,
		"whether to enable debug model.",
	)

	fs.Parse(args)
	return o
}

func main() {
	logrusutil.ComponentInit("xihe")
	log := logrus.NewEntry(logrus.StandardLogger())
	o := gatherOptions(
		flag.NewFlagSet(os.Args[0], flag.ExitOnError),
		os.Args[1:]...,
	)
	if err := o.Validate(); err != nil {
		logrus.Fatalf("Invalid options, err:%s", err.Error())
	}
	logrus.SetLevel(logrus.DebugLevel)
	if o.enableDebug {
		logrus.SetLevel(logrus.DebugLevel)
		logrus.Debug("debug enabled.")
	}

	cfg := new(config.Configuration)
	if err := config.LoadConfig(o.service.ConfigFile, cfg); err != nil {
		logrus.Fatalf("load config, err:%s", err.Error())
	}

	if err := message.Init(cfg.GetMQConfig(), log, cfg.MQ.Topics); err != nil {
		log.Fatalf("initialize mq failed, err:%v", err)
	}

	defer message.Exit(log)

	run(newHandler(cfg, log), log)
}
func newHandler(cfg *config.Configuration, log *logrus.Entry) *handler {
	competitionClient, err := client.NewCompetitionClient(cfg.Endpoint)
	if err != nil {
		logrus.Errorf("init rpc server err: %v", err)
		return nil
	}
	return &handler{
		maxRetry:  cfg.MaxRetry,
		log:       log,
		calculate: app.NewCalculateService(score.NewCalculateScore(os.Getenv("CALCULATE"))),
		evaluate:  app.NewEvaluateService(score.NewEvaluateScore(os.Getenv("EVALUATE"))),
		match:     cfg,
		cli:       competitionClient,
	}
}

func run(h *handler, log *logrus.Entry) {
	if h == nil {
		return
	}
	defer h.cli.Disconnect()
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt, syscall.SIGTERM)

	var wg sync.WaitGroup
	defer wg.Wait()

	called := false
	ctx, done := context.WithCancel(context.Background())

	defer func() {
		if !called {
			called = true
			done()
		}
	}()

	wg.Add(1)
	go func(ctx context.Context) {
		defer wg.Done()

		select {
		case <-ctx.Done():
			log.Info("receive done. exit normally")
			return

		case <-sig:
			log.Info("receive exit signal")
			done()
			called = true
			os.Exit(1)
			return
		}
	}(ctx)

	if err := message.Subscribe(ctx, h, log); err != nil {
		log.Errorf("subscribe failed, err:%v", err)
	}
}
