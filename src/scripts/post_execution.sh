#!/bin/bash

MAX_FREQ="3.6Ghz"
MIN_FREQ="1.2Ghz"

# =============================================================================

end_powerstat() {
  pwId=$(grep "PWID: " /tmp/measure_energy.tmp | awk '{print $2}')
  kill -SIGTERM "$pwId"
  kill -SIGQUIT "$pwId"
}

reset_configurations() {
  sudo cset shield --reset
  sudo cpupower --cpu all frequency-set -g ondemand  -d $MIN_FREQ -u $MAX_FREQ
}
# =============================================================================

# stop the execution of powerstat and kill the process
# check for powerstat pId from output
# terminate the process
# extract the average power and frequency information (better done in python)
end_powerstat

# restore default configurations
# unshield the shielded CPUs
# restore performance configurations
reset_configurations
