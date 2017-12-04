# Bash utility functions

This repository contains a couple of very useful bash utility functions.
Most notably argument parsing but also user dialogues, command checking
and some lab specific cuda checks.

How do you use this shell script library.
Just clone this repository into the top level of your repository.
Now just include it into any of your scripts with the following
lines, we assume that your scripts are in the directory scripts.

    # We obtain the directory of your script
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
    # Now we source our library.
    source ${SCRIPT_DIR}/../../ubash/ubash.sh || exit 1

## Contributing 

Please use the shellcheck to make sure that all files are following the
best shell practice.

```bash
sudo apt-get install shellcheck

# Check all shell script files.
shellcheck *.sh
```

## FLAGS aka argument parsing

It is often required that we have to parse arguments in a shell script.
This library will provide you with four shell script functions to 
reduce the burden of the argument parsing. For example, the help
flag is always defined and will print a usage message with all
flags and descriptions as well as default values.

In the following example we show how to use argument parsing.

    # We obtain the directory of your script
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
    # Now we source our library.
    source ${SCRIPT_DIR}/../../ubash/ubash.sh || exit 1
    
    # We define a couple of flags which we want to parse.

    # We support two types, strings and bool.
    # First two example of strings.
    # parameters are flag_name, flag_default, flag_description.
    ubash::flag_string "magic_flag" "banana" "This flag is pure magic, thus banana!"

    # or parameters are flag_name, flag_description.
    ubash::flag_string "unreal" "Dont know why but this flag is unreal."

    # Now two example of bools. For bools, true is either "" "t" or "true".
    # parameters are flag_name, flag_default, flag_description.
    ubash::flag_bool "use_me" "true" "I guess this flag should be used."

    # or parameters are flag_name, flag_description.
    ubash::flag_bool "whatever" "Some people just dont have opinions."
    
    # Finally we parse the arguments.
    ubash::parse_args "$@"

    # Now all arguments are available as 
    ubash::pp "FLAG_magic_flag = ${FLAG_magic_flag}"
    ubash::pp "FLAG_unreal = ${FLAG_unreal}"
    ubash::pp "FLAG_use_me = ${FLAG_use_me}"
    ubash::pp "FLAG_whatever = ${FLAG_whatever}"
    
    # A typical usecase of boolean flags might be as follows.
    if ${FLAG_use_me};then
        ubash::pp "We are using this for sure."
    else
        ubash::pp "So sad we cannot use what we wanted to."
    fi

Usage in binary is analogous to gflags usage:

    ./<binary> --magic_flag=apple --use_me=false


## IO

In general we often want to print things on command line. We introduce 
the function `ubash::pp` which will print with regular expressions enabled.

In order to check the operating system we set the variable `UBASH_OS` which 
will either be `linux` or `mac`.

Sometimes we want to interact with a user, often in a yes/no fashion, for that
we provide the function `ubash::user_confirm "message" "default"` which will return
the variable USER_CONFIRM_RESULT which is either y or n. 

    # We obtain the directory of your script
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
    # Now we source our library.
    source ${SCRIPT_DIR}/../../ubash/ubash.sh || exit 1
    
    # Often the user wants to know the root directory of some project
    # since we assume by convention that ubash is in the top level directory
    # we can create a variable called 
    ubash::pp "The project root = ${PROJECT_DIR}"
    # which can be very helpful for some path magic.


    # For example in case we want to ask the user to continue with the 
    # installation and per default return no.
    ubash::user_confirm "continue with installation" "n"
    
    if [[ "y" == "${USER_CONFIRM_RESULT}" ]];then
        ubash::pp "The user really wants to install our stuff"
    fi
    
    # or per default return yes.
    ubash::user_confirm "just do it" "y"
    if [[ "y" == "${USER_CONFIRM_RESULT} ]];then
        ubash::pp "We will just do it no matter what."
    else
        ubash::pp "Really sad, we had to quit."
    fi
    
    if ! ubash::command_exists docker; then
        ubash::pp "docker is required please install it."
        exit 1
    fi



## CUDA

Some people like to use a lot of parallel computation units, most often
this will require them to add cuda to their (DY)LD_LIBRARY_PATH.
We provide two functions to add the default paths for the mpi is cluster, 
ubuntu 14.04 and mac.

    # We obtain the directory of your script
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
    # Now we source our library.
    source ${SCRIPT_DIR}/../../ubash/ubash.sh || exit 1
    
    # Just add the following lines to your script to make sure that the
    # corresponding paths are exported.
    ubash::cuda_mac
    ubash::cuda_ubuntu
    ubash::cuda_cluster
