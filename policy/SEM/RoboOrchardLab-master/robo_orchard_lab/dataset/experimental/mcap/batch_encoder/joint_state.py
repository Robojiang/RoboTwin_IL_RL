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

import torch
from google.protobuf.timestamp import from_nanoseconds
from robo_orchard_schemas.sensor_msgs.JointState_pb2 import (
    JointState as PbJointState,
    # JointStateStamped as PbJointStateStamped,
    MultiJointStateStamped as PbMultiJointStateStamped,
)

from robo_orchard_lab.dataset.datatypes.joint_state import BatchJointsState
from robo_orchard_lab.dataset.experimental.mcap.batch_encoder.base import (
    McapBatchEncoder,
    McapBatchEncoderConfig,
    StampedMessage,
)

__all__ = [
    "McapBatchFromBatchJointState",
    "McapBatchFromBatchJointStateConfig",
]


def to_numpy(tensor: torch.Tensor | None):
    if tensor is None:
        return None
    return tensor.numpy(force=True)


class McapBatchFromBatchJointState(McapBatchEncoder[BatchJointsState]):
    """Convert BatchJointsState to Mcap batch messages.

    This class converts a `BatchJointsState` object into a list of
    `MultiJointStateStamped` protobuf messages, each wrapped in
    a `StampedMessage` that includes logging and publication timestamps.

    The timestamps are required in the `BatchJointsState` object.

    """

    def __init__(self, config: McapBatchFromBatchJointStateConfig):
        super().__init__()
        self._cfg = config

    def format_batch(
        self, data: BatchJointsState
    ) -> dict[str, list[StampedMessage[PbMultiJointStateStamped]]]:
        def to_pb_multi_joint_state_stamped(
            joint_state: BatchJointsState,
        ) -> list[StampedMessage[PbMultiJointStateStamped]]:
            if joint_state.timestamps is None:
                raise ValueError("timestamps is required")
            ret: list[StampedMessage[PbMultiJointStateStamped]] = []
            batch_size = joint_state.batch_size
            joint_num = joint_state.joint_num
            position = to_numpy(joint_state.position)
            velocity = to_numpy(joint_state.velocity)
            effort = to_numpy(joint_state.effort)
            names = (
                joint_state.names
                if joint_state.names is not None
                else [None] * joint_num
            )
            for i in range(batch_size):
                state_list = [
                    PbJointState(
                        frame_id=names[j],
                        name=names[j],
                        position=position[i, j]
                        if position is not None
                        else None,
                        velocity=velocity[i, j]
                        if velocity is not None
                        else None,
                        effort=effort[i, j] if effort is not None else None,
                    )
                    for j in range(joint_num)
                ]
                ret.append(
                    StampedMessage(
                        data=PbMultiJointStateStamped(
                            timestamp=from_nanoseconds(
                                joint_state.timestamps[i]
                            ),
                            states=state_list,
                        ),
                        log_time=joint_state.timestamps[i],
                        pub_time=joint_state.timestamps[i],
                    )
                )
            return ret

        return {
            self._cfg.target_topic: to_pb_multi_joint_state_stamped(data),
        }


class McapBatchFromBatchJointStateConfig(
    McapBatchEncoderConfig[McapBatchFromBatchJointState],
):
    class_type: type[McapBatchFromBatchJointState] = (
        McapBatchFromBatchJointState
    )

    target_topic: str
    """The target topic to publish the encoded batch messages."""
