# -*- coding: utf-8 -*-
"""Utility functions."""


def flatten_dict(input_dict, parent_key='', sep='.'):
    """Recursive function to convert multi-level dictionary to flat dictionary.

    Args:
        input_dict (dict): Dictionary to be flattened.
        parent_key (str): Key under which `input_dict` is stored in the higher-level dictionary.
        sep (str): Separator used for joining together the keys pointing to the lower-level object.

    """
    items = []  # list for gathering resulting key, value pairs
    for k, v in input_dict.items():
        new_key = parent_key + sep + k.replace(" ", "") if parent_key else k.replace(" ", "")
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
