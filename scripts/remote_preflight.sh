#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' 'Host OS'
cat /etc/os-release
printf '%s\n' 'GPU and driver (Isaac Sim 5.1 tested Linux driver: 580.65.06)'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
printf '%s\n' 'Available RAM and disk'
free -h
df -h .
printf '%s\n' 'Docker server'
docker version --format '{{.Server.Version}}'
printf '%s\n' 'NVIDIA container runtime GPU access'
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
