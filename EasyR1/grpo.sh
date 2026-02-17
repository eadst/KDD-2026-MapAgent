#!/bin/bash

set -x

MODEL_PATH=/root/paddlejob/workspace/env_run/lzh/models/qwen3-vl-8b/map_agent_0123_1423_merged # replace it with your local file path

python3 -m verl.trainer.main \
    config=experiments/my_config.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \

