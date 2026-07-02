#!/bin/bash

docker pull lmsysorg/sglang:dev

docker run -it --shm-size 32g --gpus all -v /data01/cache/huggingface:/root/.cache/huggingface -v /data02/<your-name>:/data --ipc=host --privileged --name <project-name>-<your-name> lmsysorg/sglang:dev /bin/zsh

docker run -it \
  --gpus all \
  --network host \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --cap-add=SYS_ADMIN \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
   --privileged \
  -e SHELL=/bin/bash \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -w /data \
  -v /data01/cache/huggingface:/root/.cache/huggingface \
  -v /data02/triplemu:/data \
  --name sglang-diffusion-ulysses \
  nvcr.io/nvidia/pytorch:25.03-py3 \
  bash
