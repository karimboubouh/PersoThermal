#!/bin/bash

source src/scripts/helpers.sh

# parse script arguments in format << --param value >>
process_arguments "$@"

# shield the selected CPU(s).
shield_cpu

# configure shielded CPU frequency
configure_cpu_performance

# run powerstat in idle state for 10min
#idle_powerstat

# run a given program for $runs times
multi_run

# reset configurations

reset_configurations
