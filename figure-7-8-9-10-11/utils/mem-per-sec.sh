#!/usr/bin/env bash

while true; do
  echo -n "$(date '+%H:%M:%S') "
  free -m | awk 'NR==2 {print $0}'
  sleep 1
done