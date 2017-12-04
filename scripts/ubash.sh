
SCRIPT_DIR_UBASH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
UBASH_DIR="${SCRIPT_DIR_UBASH}/../ubash"
UBASH_PATH="${UBASH_DIR}/ubash.sh"
if [[ ! -e "${UBASH_PATH}" ]];then
    CUR_DIR=$(pwd)
    cd ${UBASH_DIR};
    git submodule update --init 
    cd ${CUR_DIR}
fi

if [[ -e "${UBASH_PATH}" ]];then
    CUR_DIR=$(pwd)
    cd ${UBASH_DIR};
    if [[ "$(git branch)" != "master" ]];then
        git checkout master > /dev/null 2>&1
    fi
    git pull > /dev/null 2>&1
    cd ${CUR_DIR}
fi

source ${UBASH_PATH} || exit 1
