#!/bin/bash
# Author: Daniel Kappler
# Please contact daniel.kappler@gmail.com if bugs or errors are found
# in this script.

# Do not use -e since this file is supposed to be sourced.

# This script contains a bash imitation of gflags.
# Please have a look at ubash_flags_test.sh for example.
# A collection of utility functions for bash.
# This script is always assumed to be checked out in the top level
# directory of the project of choice.

# The directory of ubash.
UBASH_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"

source "${UBASH_DIR}/ubash_io.sh" || exit 1
source "${UBASH_DIR}/ubash_flags.sh" || exit 1
source "${UBASH_DIR}/ubash_cuda.sh" || exit 1
source "${UBASH_DIR}/ubash_py.sh" || exit 1
source "${UBASH_DIR}/ubash_catkin.sh" || exit 1

