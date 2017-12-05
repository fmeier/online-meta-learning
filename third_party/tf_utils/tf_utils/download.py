
import os as _os

import requests


def maybe_download(dir_path_data, file_name, source_url):
    if not _os.path.exists(dir_path_data):
        _os.makedirs(dir_path_data)
    file_path = _os.path.join(dir_path_data, file_name)
    if _os.path.exists(file_path):
        return file_path
    print('We are downloading the file {}.'.format(source_url))

    response = requests.get(source_url)
    with open(file_path, 'w') as fo:
        fo.write(response.content)
    return file_path
