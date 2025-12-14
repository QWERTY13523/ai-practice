#!/bin/bash

# 定义清理函数：当按下 Ctrl+C 时，杀死所有子进程
trap 'kill $(jobs -p); exit' SIGINT SIGTERM

echo "启动 CosyVoice..."
bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate cosyvoice && cd ~/AI-practice && python worker_cosy.py" &
PID1=$!

sleep 10

echo "启动 IndexTTS..."
bash -c "conda activate biindex && cd ~/AI-practice/index-tts && uv run worker_index.py" &
PID2=$!

sleep 10

echo "启动 Director..."
bash -c "conda activate biindex && cd ~/AI-practice && python director.py" &
PID3=$!

echo "所有服务已启动，按 Ctrl+C 停止所有服务"
wait