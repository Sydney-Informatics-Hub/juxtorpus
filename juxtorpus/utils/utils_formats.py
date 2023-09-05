""" utils

"""

from typing import Type


def format_dtm_id(clazz: Type, *args) -> str:
    """ Unify formatting of dtm names. e.g. 'class_name/arg0-arg1'
    :param clazz - the class this new dtm is generated with
    :param *args - variadic args for distinguishing information.
    """
    return f"{clazz.__name__}/{'-'.join((str(arg) for arg in args))}"


def format_meta_id(clazz: Type, *args) -> str:
    """ Unify formatting of meta ids. e.g. 'class_name_arg0-arg1."""
    return f"{clazz.__name__}_{'-'.join((str(arg) for arg in args))}"
