#!/bin/bash

# 使途：リオープンした際に最初にsourceで実行すると、手間が省ける

source ./init_python_venv.sh
PROMPT_COMMAND="history -a; $PROMPT_COMMAND"
# cd ./prototypes/kkd_lib_proto/
