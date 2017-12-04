
ubash::catkin_flags() {
    ubash::flag_bool "catkin_clean_workspace" \
                     "false" \
                     "If true, remove the build, devel, release."

    ubash::flag_string "catkin_build_type" \
                     "Release" \
                     "The release type passed to -DCMAKE_BUILD_TYPE."
}

ubash::catkin_find_workspace() {
    local NOT_DONE="true"
    local CUR_DIR=$(pwd)
    local WORKSPACE_PATH=${PROJECT_DIR}
    local USE_HOSTNAME="false"
    if [[ "$#" -eq 1 ]];then
        local parameter="$1"
        local value="${parameter%=*}"
        if [[ ${value} == "use_hostname" ]];then
            local USE_HOSTNAME="${parameter#*=}"
        else
            ubash::pp_error "We only know the parameter use_hostname=[true/false] but received ${parameter}."
        fi
    fi
    
    while [[ ${NOT_DONE} == "true" ]] ;do
        if [[ -e ${WORKSPACE_PATH}/workspace ]];then
            NOT_DONE="false"
        else
            cd ..;
            WORKSPACE_PATH=$(pwd)
            if [[ ${WORKSPACE_PATH} == "/" ]];then
                ubash::pp_error "We have not found a workspace."
                NOT_DONE="false"
            fi
        fi
    done
    local HOST_NAME=""
    if [[ "${USE_HOSTNAME}" == "true" ]];then
        HOST_NAME="_$(hostname)"
    fi

    WORKSPACE_PATH="${WORKSPACE_PATH}/workspace"
    # We define these variables to be used in the main ubash files.
    # shellcheck disable=SC2034
    UBASH_CATKIN_WORKSPACE_PATH="${WORKSPACE_PATH}"
    # We define these variables to be used in the main ubash files.
    # shellcheck disable=SC2034
    UBASH_CATKIN_DEVEL_PATH="${WORKSPACE_PATH}/devel${HOST_NAME}"
    # We define these variables to be used in the main ubash files.
    # shellcheck disable=SC2034
    UBASH_CATKIN_BUILD_PATH="${WORKSPACE_PATH}/build${HOST_NAME}"
    # We define these variables to be used in the main ubash files.
    # shellcheck disable=SC2034
    UBASH_CATKIN_INSTALL_PATH="${WORKSPACE_PATH}/install${HOST_NAME}"

    # We reset the directory to the previous directory.
    cd "${CUR_DIR}"
}


ubash::catkin_find_ros_path() {
    # We populate UBASH_ROS_PATH.
    if [[ -e "/opt/ros/indigo" ]];then
        # shellcheck disable=SC2034
        UBASH_CATKIN_ROS_PATH="/opt/ros/indigo"
    elif [[ -e "/opt/ros/groovy" ]];then
        # shellcheck disable=SC2034
        UBASH_CATKIN_ROS_PATH="/opt/ros/groovy"
    else
        ubash::pp_error "We could not find neither groovy nor indigo."
    fi
}

ubash::catkin_install_exists() {
    if [[ -e "${UBASH_INSTALL_PATH}" ]];then
        # UBASH_RETRUN is our default return value.
        # shellcheck disable=SC2034
        UBASH_RETRUN=true
    else
        # UBASH_RETRUN is our default return value.
        # shellcheck disable=SC2034
        UBASH_RETRUN=false
    fi
}

ubash::catkin_clean_workspace() {
    if ${FLAG_catkin_clean_workspace} ;then
        ubash::pp "Clean the workspace"
        ubash::check_path "${UBASH_CATKIN_WORKSPACE_PATH}"
        ubash::check_path "${UBASH_CATKIN_DEVEL_PATH}"
        ubash::check_path "${UBASH_CATKIN_BUILD_PATH}"
        ubash::check_path "${UBASH_CATKIN_INSTALL_PATH}"
        if [[ -d ${UBASH_CATKIN_DEVEL_PATH} ]];then
            ubash::user_confirm "Delete ${UBASH_CATKIN_DEVEL_PATH}" "n"
            if [[ "y" == "${USER_CONFIRM_RESULT}" ]];then
                rm -rf "${UBASH_CATKIN_DEVEL_PATH}"
            fi
        fi
        if [[ -d ${UBASH_CATKIN_BUILD_PATH} ]];then
            ubash::user_confirm "Delete ${UBASH_CATKIN_BUILD_PATH}" "n"
            if [[ "y" == "${USER_CONFIRM_RESULT}" ]];then
                rm -rf "${UBASH_CATKIN_BUILD_PATH}"
            fi
        fi
        if [[ -d ${UBASH_CATKIN_INSTALL_PATH} ]];then
            ubash::user_confirm "Delete ${UBASH_CATKIN_INSTALL_PATH}" "n"
            if [[ "y" == "${USER_CONFIRM_RESULT}" ]];then
                rm -rf "${UBASH_CATKIN_INSTALL_PATH}"
            fi
        fi
    fi
}

ubash::catkin_compile_workspace() {
    ubash::check_path "${UBASH_CATKIN_WORKSPACE_PATH}"
    ubash::check_path "${UBASH_CATKIN_DEVEL_PATH}"
    ubash::check_path "${UBASH_CATKIN_BUILD_PATH}"
    ubash::check_path "${UBASH_CATKIN_INSTALL_PATH}"
    ubash::pp "We compile our workspace ${UBASH_CATKIN_WORKSPACE_PATH}"
    ubash::pp "with build in ${UBASH_CATKIN_BUILD_PATH}"
    ubash::pp "with devel in ${UBASH_CATKIN_DEVEL_PATH}"

    local CUR_DIR="$(pwd)"

    # Determine the ros path.
    ubash::catkin_find_ros_path

    cd "${UBASH_CATKIN_WORKSPACE_PATH}"
    # We allow them to create unbound variables since we do not control it.
    set +u
    source "${UBASH_CATKIN_ROS_PATH}/setup.bash"
    set -u

    local ARGUMENTS="" 
    ARGUMENTS="--build=${UBASH_CATKIN_BUILD_PATH}"
    ARGUMENTS="${ARGUMENTS} -DCATKIN_DEVEL_PREFIX=${UBASH_CATKIN_DEVEL_PATH}"
    ARGUMENTS="${ARGUMENTS} -DCATKIN_INSTALL_PREFIX=${UBASH_CATKIN_INSTALL_PATH}"
    ARGUMENTS="${ARGUMENTS} -DCMAKE_INSTALL_PREFIX=${UBASH_CATKIN_INSTALL_PATH}"

    if [[ "${FLAG_catkin_build_type}" != "" ]];then
        ARGUMENTS="${ARGUMENTS} -DCMAKE_BUILD_TYPE=${FLAG_catkin_build_type}"
    fi

    if [[ $# -gt 0 ]];then
        # shellcheck disable=SC2124
        ARGUMENTS="${ARGUMENTS} $@"
    fi

    # We output our build command.
    set -x
    # We have to pass the unquoted string, it would be great to have a
    # different solution for this problem.
    # shellcheck disable=SC2086
    catkin_make ${ARGUMENTS}
    set +x

    cd "${CUR_DIR}"
}


ubash::catkin_install_workspace() {
    ubash::check_path "${UBASH_CATKIN_WORKSPACE_PATH}"
    ubash::check_path "${UBASH_CATKIN_DEVEL_PATH}"
    ubash::check_path "${UBASH_CATKIN_BUILD_PATH}"
    ubash::pp "We install our workspace ${UBASH_CATKIN_WORKSPACE_PATH}"
    ubash::pp "with build in ${UBASH_CATKIN_BUILD_PATH}"
    ubash::pp "with devel in ${UBASH_CATKIN_DEVEL_PATH}"
    ubash::pp "with install in ${UBASH_CATKIN_INSTALL_PATH}"

    local CUR_DIR="$(pwd)"
    cd "${UBASH_CATKIN_WORKSPACE_PATH}"
    ubash::catkin_compile_workspace "$@ install"
    cd "${CUR_DIR}"
}


ubash::catkin_source_devel() {
    if [[ -e ${UBASH_CATKIN_DEVEL_PATH} ]];then
        ubash::pp "We are sourcing the workspace"
        ubash::pp "${UBASH_CATKIN_DEVEL_PATH}/setup.bash"
        source "${UBASH_CATKIN_DEVEL_PATH}/setup.bash" || exit 1
    else
        ubash::pp_error "No workspace setup.bash at ${UBASH_CATKIN_DEVEL_PATH}."
        ubash::pp_error "Please compile the workspace."
    fi
}
