from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TensorRequest(_message.Message):
    __slots__ = ("input_data", "return_data")
    INPUT_DATA_FIELD_NUMBER: _ClassVar[int]
    RETURN_DATA_FIELD_NUMBER: _ClassVar[int]
    input_data: bytes
    return_data: ReturnData
    def __init__(self, input_data: _Optional[bytes] = ..., return_data: _Optional[_Union[ReturnData, _Mapping]] = ...) -> None: ...

class RequestResult(_message.Message):
    __slots__ = ("is_success", "description")
    IS_SUCCESS_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    is_success: bool
    description: str
    def __init__(self, is_success: bool = ..., description: _Optional[str] = ...) -> None: ...

class ReturnData(_message.Message):
    __slots__ = ("request_id", "start_node_addr")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    START_NODE_ADDR_FIELD_NUMBER: _ClassVar[int]
    request_id: int
    start_node_addr: str
    def __init__(self, request_id: _Optional[int] = ..., start_node_addr: _Optional[str] = ...) -> None: ...

class GetSavedTensorRequest(_message.Message):
    __slots__ = ("request_id",)
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: int
    def __init__(self, request_id: _Optional[int] = ...) -> None: ...

class DeviceFlops(_message.Message):
    __slots__ = ("fp32", "fp16", "int8")
    FP32_FIELD_NUMBER: _ClassVar[int]
    FP16_FIELD_NUMBER: _ClassVar[int]
    INT8_FIELD_NUMBER: _ClassVar[int]
    fp32: float
    fp16: float
    int8: float
    def __init__(self, fp32: _Optional[float] = ..., fp16: _Optional[float] = ..., int8: _Optional[float] = ...) -> None: ...

class NodeStatsResponse(_message.Message):
    __slots__ = ("model", "chip", "memory", "flops")
    MODEL_FIELD_NUMBER: _ClassVar[int]
    CHIP_FIELD_NUMBER: _ClassVar[int]
    MEMORY_FIELD_NUMBER: _ClassVar[int]
    FLOPS_FIELD_NUMBER: _ClassVar[int]
    model: str
    chip: str
    memory: int
    flops: DeviceFlops
    def __init__(self, model: _Optional[str] = ..., chip: _Optional[str] = ..., memory: _Optional[int] = ..., flops: _Optional[_Union[DeviceFlops, _Mapping]] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
