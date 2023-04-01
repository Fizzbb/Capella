FROM ubuntu:20.04

RUN apt-get update && apt-get install -y git vim

ENV DEBIAN_FRONTEND=noninteractive

# install python (3.8.10)
RUN apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 --no-cache-dir install --upgrade pip \
  && rm -rf /var/lib/apt/lists/*

RUN git clone --recurse-submodules https://github.com/nomic-ai/gpt4all.git
WORKDIR "/gpt4all"
#RUN git submodule configure &&  git submodule update
RUN python -m pip install -r requirements.txt

WORKDIR "/gpt4all/transformers"
RUN pip install -e .

WORKDIR "/gpt4all/peft"
RUN pip install -e .

#CMD ["bash", "-c", "accelerate launch --dynamo_backend=inductor --num_processes=8 --num_machines=1 --machine_rank=0 --deepspeed_multinode_launcher standard --mixed_precision=bf16  --use_deepspeed --deepspeed_config_file=configs/deepspeed/ds_config.json train.py --config configs/train/finetune-7b.yaml"]

CMD ["bash", "-c", "sleep infinity"]
