#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-/workspace}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

if [ -f /tmp/thimpc-jupyter.pid ] && kill -0 "$(cat /tmp/thimpc-jupyter.pid)" 2>/dev/null; then
    echo "JupyterLab already running on port 8888"
    exit 0
fi

nohup /usr/local/bin/jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --notebook-dir=/workspace \
    > /tmp/jupyter.log 2>&1 &

echo "$!" > /tmp/thimpc-jupyter.pid
echo "JupyterLab started on port 8888; log: /tmp/jupyter.log"
