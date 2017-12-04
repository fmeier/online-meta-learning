#!/bin/bash
# Author: Daniel Kappler
# Please contact daniel.kappler@gmail.com if bugs or errors are found
# in this script.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"

source "${SCRIPT_DIR}/ubash.sh" || exit 1

ubash::flag_script_description cat <<EOF
This is an example script on how to use the ubash flag utils.
EOF

ubash::flag_string "test_a" "A test input for a."
ubash::flag_bool "test_b" "A test input for b."
ubash::flag_string "test_c" "A test input for c."
ubash::flag_string "test_d" "hello" "A test input for c."

ubash::parse_args "$@"

echo "FLAG_test_a = $FLAG_test_a"
echo "FLAG_test_b = $FLAG_test_b"
echo "FLAG_test_c = $FLAG_test_c"
echo "FLAG_test_d = $FLAG_test_d"
