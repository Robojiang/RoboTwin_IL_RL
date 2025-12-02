# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
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

from foxglove_schemas_protobuf.FrameTransform_pb2 import (
    FrameTransform as FgFrameTransform,
)
from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion as FgQuaternion
from foxglove_schemas_protobuf.Vector3_pb2 import Vector3 as FgVector3
from google.protobuf.timestamp import from_nanoseconds

from robo_orchard_lab.dataset.datatypes import BatchFrameTransform
from robo_orchard_lab.dataset.experimental.mcap.batch_encoder.base import (
    McapBatchEncoder,
    McapBatchEncoderConfig,
    StampedMessage,
)

__all__ = [
    "McapBatchFromBatchFrameTransform",
    "McapBatchFromBatchFrameTransformConfig",
]


class McapBatchFromBatchFrameTransform(McapBatchEncoder[BatchFrameTransform]):
    def __init__(self, config: McapBatchFromBatchFrameTransformConfig):
        super().__init__()
        self._cfg = config

    def format_batch(
        self, data: BatchFrameTransform
    ) -> dict[str, list[StampedMessage[FgFrameTransform]]]:
        def to_pb_frame_transform_stamped(
            frame_transform: BatchFrameTransform,
        ) -> list[StampedMessage[FgFrameTransform]]:
            if frame_transform.timestamps is None:
                raise ValueError(
                    "BatchFrameTransform must have timestamps for conversion."
                )

            ret: list[StampedMessage[FgFrameTransform]] = []
            batch_size = frame_transform.batch_size
            xyz = frame_transform.xyz.numpy(force=True)
            quat = frame_transform.quat.numpy(force=True)

            for i in range(batch_size):
                frame_tf = FgFrameTransform(
                    timestamp=from_nanoseconds(frame_transform.timestamps[i]),
                    parent_frame_id=frame_transform.parent_frame_id,
                    child_frame_id=frame_transform.child_frame_id,
                    translation=FgVector3(
                        x=xyz[i, 0],
                        y=xyz[i, 1],
                        z=xyz[i, 2],
                    ),
                    rotation=FgQuaternion(
                        w=quat[i, 0],
                        x=quat[i, 1],
                        y=quat[i, 2],
                        z=quat[i, 3],
                    ),
                )

                # Create a stamped message with the timestamp
                stamped_msg = StampedMessage(
                    data=frame_tf,
                    log_time=frame_transform.timestamps[i],
                    pub_time=frame_transform.timestamps[i],
                )
                ret.append(stamped_msg)
            return ret

        return {self._cfg.target_topic: to_pb_frame_transform_stamped(data)}


class McapBatchFromBatchFrameTransformConfig(
    McapBatchEncoderConfig[McapBatchFromBatchFrameTransform]
):
    """Configuration for converting BatchFrameTransform to Mcap batch messages."""  # noqa: E501

    class_type: type[McapBatchFromBatchFrameTransform] = (
        McapBatchFromBatchFrameTransform
    )

    target_topic: str
    """The target topic to publish the encoded batch messages."""
