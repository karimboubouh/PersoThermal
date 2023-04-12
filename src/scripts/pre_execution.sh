#!/bin/bash

# set default arguments (to be extracted from python later)
MAX_FREQ="3.6Ghz"
AVG_FREQ="1.9Ghz"
MIN_FREQ="1.2Ghz"
cpu=0
performance="avg"

# =============================================================================
shield_cpu() {
  pid=$1
  sudo cset shield --reset
  sudo cset shield --cpu "$cpu" --kthread=on
  sudo -E env PATH="$PATH" cset shield --shield --pid "$pid"
}

configure_cpu_performance() {
  echo "SCRIPT: Using $performance performance"
  sudo cpupower --cpu all frequency-set -g userspace -d $MIN_FREQ -u $MIN_FREQ
  if [ "$performance" == "max" ]; then
    sudo cpupower --cpu "$cpu" frequency-set -g userspace -d $MAX_FREQ -u $MAX_FREQ
  elif [ "$performance" == "min" ]; then
    sudo cpupower --cpu "$cpu" frequency-set -g userspace -d $MIN_FREQ -u $MIN_FREQ
  elif [ "$performance" == "avg" ]; then
    sudo cpupower --cpu "$cpu" frequency-set -g userspace -d $AVG_FREQ -u $AVG_FREQ
  else
    sudo cpupower --cpu "$cpu" frequency-set -g ondemand -d $MIN_FREQ -u $MAX_FREQ
  fi
}

start_powerstat() {
  pid=$1
  # UUID=$(cat /dev/urandom | tr -dc 'A-Z0-9' | fold -w 4 | head -n 1)
  UUID=$RANDOM
  filename="/tmp/${UUID}_powerstat.log"
  powerstat 1 600000 -Rf >"$filename" &
  pwPid=$!
  printf "PID: %s\nFile: %s\nPWID: %s" "$pid" "$filename" "$pwPid" >/tmp/measure_energy.tmp
}

# =============================================================================

# get the current PID of the program
# shield the selected CPU(s).
# move the PID to the isolated CPU

shield_cpu $1

# configure shielded CPU frequency
configure_cpu_performance

# start monitoring power usage using powerstat
# start powerstat > log file
# get pID of powerstat and save it in output file
start_powerstat $1
