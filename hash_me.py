import hashlib


def hash_me(*args):
    """
    Only accepts single-value parameters. Does not accept arrays or lists
    """
    hash_str = ''
    for i, arg in enumerate(args):
        if i > 0:
            hash_str += '_'
        if isinstance(arg, str):
            hash_str += arg
        else:
            hash_str += f'{arg:e}'

    hash = hashlib.md5(hash_str.encode('utf-8'))
    return hash.hexdigest()
