#!/bin/bash

set -e

# get the directory of the current bash script
readonly SCRIPT_DIR="$(dirname -- "$(realpath -- "$0")")"

# 激活当前目录的 uv 虚拟环境
source ${SCRIPT_DIR}/.venv/bin/activate

# 运行 main.py
python ${SCRIPT_DIR}/main.py
