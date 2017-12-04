#!/bin/bash
# Author: Daniel Kappler
# Please contact daniel.kappler@gmail.com if bugs or errors are found
# in this script.

# This script contains a bash imitation of gflags.
# Please have a look at ubash_flags_test.sh for example.

UBASH_FLAG_SCRIPT_DESCRIPTION=""
UBASH_FLAG_NAMES=()
UBASH_FLAG_DESCRIPTIONS=()
UBASH_FLAG_DEFAULTS=()
UBASH_FLAG_TYPES=()

ubash::flag_script_description() {
    if [[ "$#" -ne 1 ]];then
        ubash::pp_error "${FUNCNAME} needs one input argument."
        exit 1
    fi
    UBASH_FLAG_SCRIPT_DESCRIPTION="$1"
}

ubash::flag_bool() {
    # A way to parse command line flags in a similar way to gflags.
    # ubash::flag_bool "flag_name" "flag_description"
    # or
    # ubash::flag_bool "flag_name" "flag_default" "flag_description"
    # An empty value for boolean flags is considered a true value.
    if [[ $# == 2 ]];then
        UBASH_FLAG_NAMES+=("$1")
        UBASH_FLAG_DESCRIPTIONS+=("$2")
        UBASH_FLAG_DEFAULTS+=("false")
        UBASH_FLAG_TYPES+=("bool")
    elif [[ $# == 3 ]];then
        UBASH_FLAG_NAMES+=("$1")
        UBASH_FLAG_DESCRIPTIONS+=("$3")
        UBASH_FLAG_DEFAULTS+=("$2")
        UBASH_FLAG_TYPES+=("bool")
    else
        ubash::pp "Either flag_name flag_descpription"
        ubash::pp "or flag_name flag_default flag_descpription are required."
        exit 1
    fi
}

ubash::flag_string() {
    # A way to parse command line flags in a similar way to gflags.
    # ubash::flag_string "flag_name" "flag_description"
    # or
    # ubash::flag_string "flag_name" "flag_default" "flag_description"
    if [[ $# == 2 ]];then
        UBASH_FLAG_NAMES+=("$1")
        UBASH_FLAG_DESCRIPTIONS+=("$2")
        UBASH_FLAG_DEFAULTS+=("")
        UBASH_FLAG_TYPES+=("string")
    elif [[ $# == 3 ]];then
        UBASH_FLAG_NAMES+=("$1")
        UBASH_FLAG_DESCRIPTIONS+=("$3")
        UBASH_FLAG_DEFAULTS+=("$2")
        UBASH_FLAG_TYPES+=("string")
    else
        ubash::pp "Either flag_name flag_descpription"
        ubash::pp "or flag_name flag_default flag_descpription are required."
        exit 1
    fi
}

# We always add a help flag to show all flags.
ubash::flag_bool "help" "Print usage message."

ubash::parse_args() {
    # We allow unbound variables in this function.
    set +u
    # Please call this function always with "$@" as argument to parse
    # all user inputs.

    # We first set all default arguments.
    local arraylength=${#UBASH_FLAG_NAMES[@]}
    for (( i=1; i< arraylength +1; i++ ));do
        # TODO(dkappler): figure out the proper way of doing this
        # shellcheck disable=SC2059
        printf -v "FLAG_${UBASH_FLAG_NAMES[$i-1]}" \
               "${UBASH_FLAG_DEFAULTS[$i-1]}"
    done

    while [[ $# -gt 0 ]];do   
        # We iterate through all command line arguments.

        local found_flag=false
        local flag_name=""
        local flag_value=""
        local key="$1"
        local value="$2"


        # We iterate through all defined flags.
        for (( i=1; i< arraylength +1; i++ ));do

            local flag="${UBASH_FLAG_NAMES[$i-1]}"
            local flag_type="${UBASH_FLAG_TYPES[$i-1]}"

            # Checking if the flag has been found.
            if [[ "$key" == "--$flag" ]];then
                found_flag=true
                flag_name="$flag"
                if [[ "$flag_type" == "bool" ]];then
                   if [[ "$value" == "t" ]];then
                       flag_value=true
                   elif [[ "$value" == "true" ]];then
                       flag_value=true
                   elif [[ "$value" == "" ]];then
                       flag_value=true
                   elif [[ "${value%-*}" == "-" ]];then
                       # Already a new flag.
                       flag_value=true
                   else
                       flag_value=false
                   fi
                else
                    flag_value="$value"
                fi

                # Once we have to shift for sure the second time for
                # all non empty strings only.
                shift
                if [[ "${value%-*}" == "-" ]];then
                    continue
                elif [[ "$value" != "" ]];then
                    shift
                fi
            fi
        done

        # If the flag has not been found we check if --flag_name= is an option.
        if ! ${found_flag} ;then

            # Some shell subtring magic to split before and after =.
            value="${key#*=}"
            key="${key%=*}"

            for (( i=1; i< arraylength +1; i++ ));do
                flag="${UBASH_FLAG_NAMES[$i-1]}"
                flag_type="${UBASH_FLAG_TYPES[$i-1]}"
                if [[ "$key" == "--$flag" ]];then
                    found_flag=true
                    flag_name="$flag"
                    if [[ "$flag_type" == "bool" ]];then
                        if [[ "$value" == "t" ]];then
                            flag_value=true
                        elif [[ "$value" == "true" ]];then
                            flag_value=true
                        elif [[ "$value" == "" ]];then
                            flag_value=true
                        else
                            flag_value=false
                        fi
                    else
                        flag_value="$value"
                    fi

                    # We only have to shift once since the argument was one
                    # long string.
                    shift
                fi
            done

        fi

        if ${found_flag} ;then
            # TODO(dkappler): figure out the proper way of doing this
            # shellcheck disable=SC2059
            printf -v "FLAG_${flag_name}" "$flag_value"
        else
            ubash::pp "We ignore input $key, $value."
            shift
        fi

    done

    # If the user has specified the help flag we will print the argument
    # message.
    if ${FLAG_help} ;then
        ubash::print_usage
        exit 0
    fi

    # We disallow unbound variables in this function.
    set -u
}

ubash::print_usage() {
    ubash::pp "\nScript name: $0"
    ubash::pp "Description: ${UBASH_FLAG_SCRIPT_DESCRIPTION}\n"
    ubash::pp "Flags:\n"
    local arraylength=${#UBASH_FLAG_NAMES[@]}
    for (( i=1; i< arraylength + 1; i++ ));do
        ubash::pp "--${UBASH_FLAG_NAMES[$i-1]} [default=${UBASH_FLAG_DEFAULTS[$i-1]}] ${UBASH_FLAG_DESCRIPTIONS[$i-1]}\n"
    done
}
