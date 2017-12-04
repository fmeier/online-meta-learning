#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
source "${SCRIPT_DIR}/ubash.sh" || exit 1

ubash::pp "Create the virtual environment."
ubash::create_virtualenv
ubash::source_virtualenv

# We have to do this first since it will otherwise overwrite our tf version.
# cd ${PROJECT_DIR}/third_party/tf_utils
# ${VPY_BIN} setup.py develop

ubash::user_confirm ">> Upgrade dependencies?" "n"
if [[ "y" == "${USER_CONFIRM_RESULT}" ]];then
    ${VPY_PIP} install --upgrade pip
    ${VPY_PIP} install --upgrade \
               scipy \
               requests \
               jupyter \
               requests \
               pyopenssl \
               pyasn1 \
               ndg-httpsclient \
               ipython \
               pyyaml \
               ipdb \
               h5py \
               matplotlib \
               jinja2 \
               autopep8 \
               easydict \
               progressbar2 \
               bokeh \
               dill
fi

ubash::user_confirm ">> Install tensorflow GPU?" "n"
if [[ "y" == "${USER_CONFIRM_RESULT}" ]];then
    ubash::pp "Please install cuda 8 nvidia!"
    ubash::pp "Please install cudnn 5.1 from nvidia!"
    ubash::pp "Notice, symbolic links for libcudnn.dylib and libcuda.dylib have to be added."
    ${VPY_PIP} install --ignore-installed --upgrade tf-nightly-gpu
fi


ubash::user_confirm ">> Install tensorflow CPU?" "n"
if [[ "y" == "${USER_CONFIRM_RESULT}" ]];then
    ${VPY_PIP} install --ignore-installed --upgrade tf-nightly
fi

ubash::user_confirm ">> Install pytorch GPU?" "n"
if [[ "y" == "${USER_CONFIRM_RESULT}" ]];then
    ubash::pp "Please install cuda 8 nvidia!"
    ubash::pp "Please install cudnn 5.1 from nvidia!"
    ubash::pp "Notice, symbolic links for libcudnn.dylib and libcuda.dylib have to be added."
    ${VPY_PIP} install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl 
    ${VPY_PIP} install torchvision
fi


ubash::user_confirm ">> Install pytorch CPU?" "n"
if [[ "y" == "${USER_CONFIRM_RESULT}" ]];then
    ${VPY_PIP} install http://download.pytorch.org/whl/cu75/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl 
    ${VPY_PIP} install torchvision
fi


cd ${PROJECT_DIR}
${VPY_BIN} setup.py develop

