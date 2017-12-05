
import os
import glob
import re

CLASS_TYPE = 'class_type'


def convert_camel_to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def get_all_subclasses(class_type):
    subclasses = {}
    queue = [class_type]
    while queue:
        base = queue.pop()
        for child in base.__subclasses__():
            if child in subclasses:
                continue
            name = convert_camel_to_snake_case(child.__name__)
            if name in subclasses:
                raise Exception(
                    'We already have a class registered with the name {}.'
                    'Please rename your class {} or '
                    'the other class {}.'.format(
                        name, child.__file__, subclasses[name].__file__))
            subclasses[convert_camel_to_snake_case(child.__name__)] = child
            queue.append(child)
    return subclasses


def create_from_params(cls, key_class_type, params, **kwargs):
    if key_class_type not in params.keys():
        raise Exception('The default {} is not available [{}]'.format(
            key_class_type, ','.join(params.keys())))

    if CLASS_TYPE not in params[key_class_type].keys():
        raise Exception('The ctype is not available [{}]'.format(
            ','.join(params[key_class_type].keys())))

    subclasses = get_all_subclasses(cls)

    if params[key_class_type].class_type not in subclasses:
        raise Exception('The {} is not available [{}]'.format(
            params[key_class_type].class_type, ','.join(
                sorted(subclasses.keys()))))

    return subclasses[params[key_class_type].class_type].create_from_params(
        params[key_class_type], **kwargs)


def create_all(file_path):
    modules = glob.glob(os.path.dirname(file_path) + "/*.py")
    result = []

    for module in modules:
        if not os.path.isfile(module):
            continue
        basename, ext = os.path.splitext(module)
        if ext != '.py':
            continue
        result.append(basename)
    return result
