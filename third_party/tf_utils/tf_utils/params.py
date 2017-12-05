
import easydict

from tf_utils import utils as _utils


def load(file_path):
    params_dict = _utils.yaml_load(file_path)
    return easydict.EasyDict(params_dict)
