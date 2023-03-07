FROM golang:latest as BUILDER

# build binary
RUN mkdir -p /go/src/openeuler/go-py-message

COPY . /go/src/openeuler/go-py-message

RUN go env -w GOPROXY=https://goproxy.cn,direct

RUN cd /go/src/openeuler/go-py-message && go mod tidy && CGO_ENABLED=1 go build -v -o ./go-py-message .

# copy binary config and utils
FROM openeuler/openeuler:22.03

RUN yum update -y && yum install -y python3 && yum install -y python3-pip

RUN mkdir -p /opt/app/go-py-message/py/data

COPY ./py /opt/app/go-py-message/py

RUN chmod 755 -R /opt/app/go-py-message/py

ENV EVALUATE /opt/app/go-py-message/py/evaluate.py
ENV CALCULATE /opt/app/go-py-message/py/calculate_fid.py
ENV UPLOAD /opt/app/go-py-message/py/data/

RUN pip install -r /opt/app/go-py-message/py/requirements.txt

COPY --from=BUILDER /go/src/openeuler/go-py-message/go-py-message /opt/app/go-py-message

WORKDIR /opt/app/go-py-message/

ENTRYPOINT ["/opt/app/go-py-message/go-py-message"]