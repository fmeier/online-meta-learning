import time
import os
import yaml
import pickle as pkl


def dict_prune_by_prefix(data_dict, prefixes):
    result = {}
    for key, value in data_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                result[key] = value
                continue
    return result


class Timer(object):

    def __init__(self, comment):
        self._comment = comment

    def __enter__(self):
        self.__start = time.time()

    def __exit__(self, type, value, traceback):
        self.__finish = time.time()
        print(
            self._comment,
            "duration in seconds:",
            self.duration_in_seconds())

    def duration_in_seconds(self):
        return self.__finish - self.__start


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        print(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' "
                  "(or 'y' or 'n').\n")


def get_cache_path(dir_or_file_path):
    if os.path.isdir(dir_or_file_path):
        return os.path.join(dir_or_file_path, 'cache.pkl')
    return dir_or_file_path + '.cache.pkl'


def is_cached(dir_or_file_path):
    if os.path.exists(get_cache_path(dir_or_file_path)):
        return True
    return False


def save_cache(dir_or_file_path, data):
    pkl_save(get_cache_path(dir_or_file_path), data)


def load_cache(dir_or_file_path):
    return pkl_load(get_cache_path(dir_or_file_path))


def _load(load_fn, file_path):
    if not os.path.exists(file_path):
        raise Exception('No file exists at {}.'.format(file_path))
    with open(file_path, 'r') as fi:
        return load_fn(fi)


def yaml_load(file_path):
    return _load(yaml.load, file_path)


def yaml_save(file_path, data, check_overwrite=False, check_create=False):
    def save_fn(data, file_obj):
        yaml.dump(data, file_obj)
    _save(yaml.dump, file_path, data, check_overwrite, check_create)


def pkl_load(file_path):
    return _load(pkl.load, file_path)


def pkl_save(file_path, data, check_overwrite=False, check_create=False):
    def save_fn(data, file_obj):
        pkl.dump(data, file_obj)
    _save(save_fn, file_path, data, check_overwrite, check_create)


def _save(save_fn, file_path, data, check_overwrite=False, check_create=False):
    if os.path.exists(file_path):
        if check_overwrite and not query_yes_no(
                'File {} exists, do you want to overwrite?'.format(file_path),
                'yes'):
            return
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        if check_create and not query_yes_no(
                'Output directory {} does not exist, '
                'do you want to create it?'.format(dir_path),
                'yes'):
            return
        os.makedirs(dir_path)
    with open(file_path, 'w') as fo:
        save_fn(data, fo)


def touch(file_path, times=None):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(file_path, 'a'):
        os.utime(file_path, times)
