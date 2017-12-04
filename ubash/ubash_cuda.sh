#!/bin/bash

# Author: Daniel Kappler
# Please contact daniel.kappler@gmail.com if bugs or errors are found
# in this script.

ubash::cuda_cluster() {
    if [[ ${UBASH_OS} == "linux" ]];then
        local LPATH="/lustre/shared/caffe_shared/cuda_stuff/cuda-8.0.44/lib64"
        if [[ -d ${LPATH} ]];then
            ubash::pp "cuda cluster ${LPATH}"
            export LD_LIBRARY_PATH=${LPATH}:$LD_LIBRARY_PATH
        fi

        local LPATH="/usr/local/cudnn-5.1/lib64"
        if [[ -d ${LPATH} ]];then
            ubash::pp "cudnn cluster ${LPATH}"
            export LD_LIBRARY_PATH=${LPATH}:$LD_LIBRARY_PATH
        fi
    fi
}

ubash::cuda_ubuntu() {
    if [[ ${UBASH_OS} == "linux" ]];then
        local LPATH="/usr/local/cuda/lib64"
        if [[ -d ${LPATH} ]];then
            ubash::pp "cuda ubunut ${LPATH}"
            export LD_LIBRARY_PATH=${LPATH}:$LD_LIBRARY_PATH
        fi
    fi
}


ubash::cuda_mac() {
    if [[ ${UBASH_OS} == "mac" ]];then
        local LPATH="/usr/local/cuda/lib64"
        if [[ -d ${LPATH} ]];then
            ubash::pp "cuda mac ${LPATH}"
            export LD_LIBRARY_PATH=${LPATH}:$LD_LIBRARY_PATH
            export DYLD_LIBRARY_PATH=${LPATH}:$DYLD_LIBRARY_PATH
        fi
    fi
}


ubash::check_computation_device() {
    if [[ "$1" == "cpu" ]];then
        # We use this as an return value.
        # shellcheck disable=SC2034
        USE_GPU="f"
    elif [[ "$1" == "gpu" ]];then
        # We use this as an return value.
        # shellcheck disable=SC2034
        USE_GPU="t"
    else
        echo "$1 not in [cpu, gpu]."
        exit
    fi
}
