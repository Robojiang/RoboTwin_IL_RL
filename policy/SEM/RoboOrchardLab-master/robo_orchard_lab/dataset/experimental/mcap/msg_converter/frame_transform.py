# Project RoboOrchard
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

from __future__ import annotations

import torch
from foxglove_schemas_protobuf.FrameTransform_pb2 import (
    FrameTransform as FgFrameTransform,
)
from robo_orchard_core.utils.torch_utils import dtype_str2torch

from robo_orchard_lab.dataset.datatypes.geometry import BatchFrameTransform
from robo_orchard_lab.dataset.experimental.mcap.msg_converter.base import (
    MessageConverterConfig,
    MessageConverterStateless,
    TensorTargetConfigMixin,
)

__all__ = [
    "BatchFrameTransform",
    "ToBatchFrameTransform",
    "ToBatchFrameTransformConfig",
]


class ToBatchFrameTransform(
    MessageConverterStateless[
        FgFrameTransform | list[FgFrameTransform],
        BatchFrameTransform,
    ]
):
    """Convert a Foxglove FrameTransform message to a FrameTransform Type."""

    def __init__(
        self,
        cfg: ToBatchFrameTransformConfig,
    ):
        self._cfg = cfg
        self._dtype = dtype_str2torch(cfg.dtype)

    def convert(
        self, src: FgFrameTransform | list[FgFrameTransform]
    ) -> BatchFrameTransform:
        if not isinstance(src, list):
            tf_trans = torch.tensor(
                [src.translation.x, src.translation.y, src.translation.z],
                dtype=self._dtype,
                device=self._cfg.device,
            )
            tf_rot = torch.tensor(
                [
                    src.rotation.w,
                    src.rotation.x,
                    src.rotation.y,
                    src.rotation.z,
                ],
                dtype=self._dtype,
                device=self._cfg.device,
            )

            return BatchFrameTransform(
                child_frame_id=src.child_frame_id,
                parent_frame_id=src.parent_frame_id,
                xyz=tf_trans.to(device=self._cfg.device),
                quat=tf_rot.to(device=self._cfg.device),
                timestamps=[src.timestamp.ToNanoseconds()],
            )
        else:
            assert len(src) > 0, "List of FrameTransform cannot be empty."
            tf_trans = torch.zeros(
                (len(src), 3),
                dtype=self._dtype,
            )
            tf_rot = torch.zeros(
                (len(src), 4),
                dtype=self._dtype,
            )
            for i, tf in enumerate(src):
                tf_trans[i, :] = torch.tensor(
                    [tf.translation.x, tf.translation.y, tf.translation.z],
                    dtype=self._dtype,
                )
                tf_rot[i, :] = torch.tensor(
                    [
                        tf.rotation.w,
                        tf.rotation.x,
                        tf.rotation.y,
                        tf.rotation.z,
                    ],
                    dtype=self._dtype,
                )
            return BatchFrameTransform(
                child_frame_id=src[0].child_frame_id,
                parent_frame_id=src[0].parent_frame_id,
                xyz=tf_trans.to(device=self._cfg.device),
                quat=tf_rot.to(device=self._cfg.device),
                timestamps=[tf.timestamp.ToNanoseconds() for tf in src],
            )


class ToBatchFrameTransformConfig(
    MessageConverterConfig[ToBatchFrameTransform],
    TensorTargetConfigMixin[ToBatchFrameTransform],
):
    class_type: type[ToBatchFrameTransform] = ToBatchFrameTransform
