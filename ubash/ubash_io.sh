#!/bin/bash
# Author: Daniel Kappler
# Please contact daniel.kappler@gmail.com if bugs or errors are found
# in this script.

# The project dir, since we assume we are checked out in the top level dir.
# We define these variables to be used in the main ubash files.
# shellcheck disable=SC2034
PROJECT_DIR="${UBASH_DIR}/.."


# We define these variables to be used in the main ubash files.
# shellcheck disable=SC2034
UBASH_OS="linux"
if [[ $OSTYPE == darwin* ]];then
    # We define these variables to be used in the main ubash files.
    # shellcheck disable=SC2034
    UBASH_OS="mac"
fi

UBASH_ERROR_COLOR='\033[0;31m'
UBASH_NORMAL_COLOR='\033[0m'

ubash::pp_line() {
    if [[ "$#" -ne 1 ]];then
        ubash::pp_error "${FUNCNAME} needs a one character input arg."
        exit 1
    fi
    local USER_SYMBOL="$1"
    # Print a full line of the input character/string.
    for ((i=0;i<80;++i)); do
        echo -n "${USER_SYMBOL}";
    done
    echo
}

ubash::pp_banner() {
    if [[ "$#" -lt 1 ]];then
        ubash::pp_error "${FUNCNAME} message [separation_char]."
        ubash::pp_error "You called with $#: ${FUNCNAME} $*."
        exit 1
    fi
    if [[ "$#" -gt 2 ]];then
        ubash::pp_error "${FUNCNAME} message [separation_char]."
        ubash::pp_error "You called with $#: ${FUNCNAME} $*."
        exit 1
    fi
    local USER_MESSAGE="$1"
    if [[ "$#" -eq 2 ]];then
        local USER_SYMBOL="$2"
    else
        local USER_SYMBOL="-"
    fi
    ubash::pp_new_line
    ubash::pp_line "${USER_SYMBOL}"
    ubash::pp "${USER_SYMBOL}${USER_SYMBOL} ${USER_MESSAGE}"
    ubash::pp_line "${USER_SYMBOL}"
    ubash::pp_new_line
}

ubash::pp_new_line() {
    ubash::pp ""
}

ubash::pp() {
    ubash::print "$1"
}

ubash::print() {
    local USER_MESSAGE="$1"
    echo -e "${USER_MESSAGE}"
}

ubash::pp_error() {
    ubash::print_error "$1"
}

ubash::print_error() {
    echo -e "${UBASH_ERROR_COLOR}$1${UBASH_NORMAL_COLOR}"
}


ubash::command_exists() {
	command -v "$@" > /dev/null 2>&1
}

ubash::command_success() {
	"$@" > /dev/null 2>&1
}

ubash::check_path() {
    local USER_PATH="$1"
    if [[ "${USER_PATH}" == "/" ]];then
        ubash::pp_error "We check for root and you are using it ${USER_PATH}."
        exit 1
    fi
    if [[ "${USER_PATH}" == "" ]];then
        ubash::pp_error "We check for empty and you are using it ${USER_PATH}."
        exit 1
    fi
}

ubash::user_confirm() {
    local NOT_FINISHED=true
    while ${NOT_FINISHED} ;do
        if [[ $2 == 'y' ]]; then
            echo -n "$1 [(y)/n]: "
        elif [[ $2 == 'n' ]]; then
            echo -n "$1 [y/(n)]: "
        else
            echo -n "$1 [y/n] (default $2): "
        fi
        read USER_INPUT;
        USER_INPUT=$(echo $USER_INPUT | awk '{print tolower($0)}')
        if [[ "y" == "${USER_INPUT}" ]];then
            # We use this as a return value.
            # shellcheck disable=SC2034
            USER_CONFIRM_RESULT="y";
            NOT_FINISHED=false;
        elif [[ "n" == "${USER_INPUT}" ]];then
            # We use this as a return value.
            # shellcheck disable=SC2034
            USER_CONFIRM_RESULT="n";
            NOT_FINISHED=false;
        elif [[ "" == "${USER_INPUT}" ]];then
            # We use this as a return value.
            # shellcheck disable=SC2034
            USER_CONFIRM_RESULT="$2";
            NOT_FINISHED=false;
        else
            echo
            ubash::pp "WARNING -- Unrecognized input: ${USER_INPUT}"
            ubash::pp "Please enter 'y', 'n', or <return> (for the default ($2))"
            echo
        fi
    done
}
