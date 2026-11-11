"""Compatibility wrapper for legacy ``dataset: pic`` configurations.

The project now trains on raw 1-D signals. No image or time-frequency data is
loaded here.
"""

from .raw_signal import get_raw_signal


def get_pic(args, name="pic", data_dir="./data", include_lb_to_ulb=False, return_test=False):
    return get_raw_signal(
        args,
        name=name,
        data_dir=data_dir,
        include_lb_to_ulb=include_lb_to_ulb,
        return_test=return_test,
    )
