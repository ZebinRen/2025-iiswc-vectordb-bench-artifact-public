#!/usr/bin/env bash

while true; do
  echo -n "$(date '+%H:%M:%S') "
  grep -w nr_file_pages /proc/vmstat
  sleep 1
done