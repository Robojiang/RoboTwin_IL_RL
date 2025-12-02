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

import time
from abc import ABCMeta, abstractmethod
from typing import Any, Optional

from google.protobuf.message import Message as ProtobufMessage
from mcap_protobuf.schema import build_file_descriptor_set
from typing_extensions import Self

from mcap.records import (
    Channel,
    Message as McapMessage,
    Schema,
)
from robo_orchard_lab.dataset.experimental.mcap.messages import (
    McapMessageTuple,
)

__all__ = ["McapEncoderContext", "McapProtobufEncoder"]


class McapEncoderContext(metaclass=ABCMeta):
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError(
            "reset() method is not implemented in the encoder mixin."
        )

    def __enter__(self) -> Self:
        """Enter the context manager."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_value: Optional[BaseException],
        traceback: Optional[Any],
    ) -> None:
        """Exit the context manager."""
        self.reset()
        if exc_type is not None and exc_value is not None:
            raise exc_value

    def encode_message(
        self,
        topic: str,
        msg: Any,
        log_time: Optional[int] = None,
        pub_time: Optional[int] = None,
        sequence: int = 0,
    ) -> McapMessageTuple:
        """Encode a message to MCAP format.

        Args:
            topic (str): The topic name of the message.
            msg (Any): The message data to encode.

        Returns:
            McapMessageTuple: The encoded message in MCAP format.
        """
        raise NotImplementedError(
            "encode_message() method is not implemented in the encoder mixin."
        )


class McapProtobufEncoder(McapEncoderContext):
    """Encoder for Protobuf messages in MCAP format.

    This encoder uses the Protobuf schema to encode messages into
    MCAP format. It requires the Protobuf schema to be provided during
    initialization.
    """

    def __init__(self):
        self._schemas: list[Schema] = []
        self._channels: list[Channel] = []
        self._topic2channel_id: dict[str, int] = {}
        self._topic2schema: dict[str, tuple[int, str]] = {}

    def reset(self) -> None:
        self._schemas.clear()
        self._channels.clear()
        self._topic2channel_id.clear()
        self._topic2schema.clear()

    def _register_schema(self, msg: ProtobufMessage):
        fd_set = build_file_descriptor_set(type(msg))
        schema = Schema(
            id=len(self._schemas),
            data=fd_set.SerializeToString(),
            name=type(msg).DESCRIPTOR.full_name,
            encoding="protobuf",
        )
        self._schemas.append(schema)
        return schema.id

    def encode_message(
        self,
        topic: str,
        msg: ProtobufMessage,
        log_time: Optional[int] = None,
        pub_time: Optional[int] = None,
        sequence: int = 0,
    ) -> McapMessageTuple:
        msg_typename: str = type(msg).DESCRIPTOR.full_name
        if topic in self._topic2channel_id:
            channel_id = self._topic2channel_id[topic]
            schema_id, schema_name = self._topic2schema[topic]
            if msg_typename != schema_name:
                raise ValueError(
                    f"Topic '{topic}' has type {schema_name}, "
                    f"cannot encode a {msg_typename}"
                )
        else:
            schema_id = self._register_schema(msg)
            self._topic2schema[topic] = (schema_id, msg_typename)
            channel = Channel(
                id=len(self._channels),
                topic=topic,
                message_encoding="protobuf",
                schema_id=schema_id,
                metadata={},
            )
            self._channels.append(channel)
            self._topic2channel_id[topic] = channel.id
            channel_id = channel.id

        if log_time is None:
            log_time = time.time_ns()
        if pub_time is None:
            pub_time = time.time_ns()

        return McapMessageTuple(
            schema=self._schemas[schema_id],
            channel=self._channels[channel_id],
            message=McapMessage(
                channel_id=channel_id,
                log_time=log_time,
                data=msg.SerializeToString(),
                publish_time=pub_time,
                sequence=sequence,
            ),
        )
