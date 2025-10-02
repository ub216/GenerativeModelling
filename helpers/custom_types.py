from typing import TYPE_CHECKING, Literal, Type

if TYPE_CHECKING:
    from models.base_model import BaseModel

GenBaseModel = Type["BaseModel"]
DeviceType = Literal["cpu", "cuda"]
ReductionType = Literal["mean", "sum"]
