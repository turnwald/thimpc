#!/bin/bash
nohup /usr/local/bin/jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/workspace > /tmp/jupyter.log 2>&1 &
