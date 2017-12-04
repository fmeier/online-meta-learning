# Author: Daniel Kappler
# Please contact daniel.kappler@gmail.com if bugs or errors are found
# in this script.

# We define these variables to be used in the main ubash files.

ubash::python_flags() {
    ubash::flag_bool "use_vpy" \
                     "false" \
                     "If true, we use our virtual python environment."
}

# shellcheck disable=SC2034
VPY_DIR="${PROJECT_DIR}/vpy"
# shellcheck disable=SC2034
VPY_BIN="${VPY_DIR}/bin/python"
# shellcheck disable=SC2034
VPY_IBIN="${VPY_DIR}/bin/ipython"
# shellcheck disable=SC2034
VPY_PIP="${VPY_DIR}/bin/pip"

ubash::get_ipython() {
    if ${FLAG_use_vpy};then
        echo "${VPY_IBIN}"
    else
        echo "ipython"
    fi
}

ubash::get_python() {
    if ${FLAG_use_vpy};then
        echo "${VPY_BIN}"
    else
        echo "python"
    fi
}

ubash::get_pip() {
    if ${FLAG_use_vpy};then
        echo "${VPY_PIP}"
    else
        echo "sudo pip"
    fi
}

ubash::create_virtualenv() {
    if [[ ! -e "${VPY_DIR}" ]];then
        ubash::pp "# We setup a virtual environemnt ${VPY_DIR}."
        if ! ubash::command_exists virtualenv;then
            ubash::pp "# We install virtualenv!"
            sudo pip install virtualenv 
        fi
        virtualenv "${VPY_DIR}" --clear
        virtualenv "${VPY_DIR}" --relocatable
    fi
}

ubash::source_virtualenv() {
    if [[ ! -e "${VPY_DIR}" ]];then
        ubash::pp "# No virtual environemnt ${VPY_DIR}."
        exit 1
    fi

    source "${VPY_DIR}/bin/activate"
}

ubash::autopep8() {
    if [[ "$#" -ne 1 ]];then
        ubash::pp "# ubash::autopep8 needs one argument the root directory."
    fi
    if ! ubash::command_exists autopep8; then
        ubash::pp "# We need autopep8."
        ubash::pp "# Please install autopep by calling: "
        ubash::pp "# sudo pip install autopep8;"
    else
        ubash::user_confirm "Run autopep for directory $1 and below?" "n"
        if [[ "y" == "${USER_CONFIRM_RESULT}" ]];then
            set -x
            find "$1" -name '*.py' -exec autopep8 --in-place -a '{}' \;
        fi
    fi
}
