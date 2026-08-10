"""nnU-Net network construction helper.

Adapted from nnUNetv2.utilities.get_network_from_plans (nnU-Net v2.8.0).
The architecture arguments are supplied by the packaged plans.json.
"""

from __future__ import annotations

import pydoc
import warnings
from typing import Optional

from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class


def get_network_from_plans(
    arch_class_name,
    arch_kwargs,
    arch_kwargs_req_import,
    input_channels,
    output_channels,
    allow_init: bool = True,
    deep_supervision: Optional[bool] = None,
):
    """Instantiate the architecture encoded in an nnU-Net plans file."""
    architecture_kwargs = dict(**arch_kwargs)
    for required_import in arch_kwargs_req_import:
        if architecture_kwargs[required_import] is not None:
            architecture_kwargs[required_import] = pydoc.locate(
                architecture_kwargs[required_import]
            )

    network_class = pydoc.locate(arch_class_name)
    if network_class is None:
        warnings.warn(
            f"Network class {arch_class_name} was not found directly; "
            "searching dynamic_network_architectures.architectures."
        )
        import dynamic_network_architectures

        network_class = recursive_find_python_class(
            join(
                dynamic_network_architectures.__path__[0],
                "architectures",
            ),
            arch_class_name.split(".")[-1],
            "dynamic_network_architectures.architectures",
        )
    if network_class is None:
        raise ImportError(
            f"Could not locate network class {arch_class_name!r}"
        )

    if deep_supervision is not None:
        architecture_kwargs["deep_supervision"] = deep_supervision
    network = network_class(
        input_channels=input_channels,
        num_classes=output_channels,
        **architecture_kwargs,
    )
    if hasattr(network, "initialize") and allow_init:
        network.apply(network.initialize)
    return network


def build_network_from_plans_json(
    plans: dict,
    configuration: str = "2d",
    input_channels: int = 1,
    output_channels: int = 2,
    deep_supervision: bool = False,
):
    """Build Dataset351's network directly from plans.json."""
    architecture = plans["configurations"][configuration]["architecture"]
    return get_network_from_plans(
        arch_class_name=architecture["network_class_name"],
        arch_kwargs=architecture["arch_kwargs"],
        arch_kwargs_req_import=architecture["_kw_requires_import"],
        input_channels=input_channels,
        output_channels=output_channels,
        deep_supervision=deep_supervision,
    )

