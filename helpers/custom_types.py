from typing import TYPE_CHECKING, Literal, Type

if TYPE_CHECKING:
    from models.base_model import BaseModel
    from models.ema import EMAModel

GenBaseModel = Type["BaseModel"]
GenEMAModel = Type["EMAModel"]
DeviceType = Literal["cpu", "cuda"]
ReductionType = Literal["mean", "sum"]
