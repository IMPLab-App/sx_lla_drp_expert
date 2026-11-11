# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from importlib import import_module


def _load_dataset_entry(module_name, function_name):
    try:
        module = import_module(f"{__name__}.{module_name}")
        return getattr(module, function_name)
    except ModuleNotFoundError as exc:
        def _missing_dataset(*args, **kwargs):
            raise ModuleNotFoundError(
                f"Optional dependency for dataset '{module_name}' is missing: {exc.name}"
            ) from exc

        return _missing_dataset


get_svhn = _load_dataset_entry("svhn", "get_svhn")
get_cifar = _load_dataset_entry("cifar", "get_cifar")
get_pic = _load_dataset_entry("pic", "get_pic")
get_raw_signal = _load_dataset_entry("raw_signal", "get_raw_signal")
get_stl10 = _load_dataset_entry("stl10", "get_stl10")
get_semi_aves = _load_dataset_entry("aves", "get_semi_aves")
get_food101 = _load_dataset_entry("food101", "get_food101")
get_eurosat = _load_dataset_entry("eurosat", "get_eurosat")
get_imagenet = _load_dataset_entry("imagenet", "get_imagenet")
get_medmnist = _load_dataset_entry("medmnist", "get_medmnist")
