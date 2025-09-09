from abc import *


class AbsJoiner(metaclass=ABCMeta):
    @abstractmethod
    def append_context(
        self,
        start_window: int,
        start_second: float,
        end_second: float,
        recognized: str,
        min_avg: float,
        segment_index: int,
    ):
        pass

    @abstractmethod
    def get_string(self) -> str:
        pass
