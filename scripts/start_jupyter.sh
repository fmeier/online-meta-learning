#!/bin/bash

# Include our base tools.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
source "${SCRIPT_DIR}/ubash.sh" || exit 1

DESCRIPTION=$(cat << 'EOF'

This script is used to start a jupyter notebook with the local python
environment. All arguments to this script are passed through to
jupyter notebook.

EOF
)

ubash::flag_script_description "${DESCRIPTION}"

ubash::parse_args "$@"

cd ${PROJECT_DIR}
${VPY_DIR}/bin/jupyter notebook --ip 0.0.0.0 "$@"
