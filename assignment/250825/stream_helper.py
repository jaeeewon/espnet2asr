import numpy as np
from typing import List, Tuple
import asyncio
import time
from espnet2.bin.asr_inference import Speech2Text
from dataclasses import dataclass


# ex. [ ( (0, 3), "hello", 0.82) ] | ( (from, to), from_model, exe_time )[]
@dataclass
class Status:
    window: Tuple[int, int]
    output: str
    exec_sec: float

    def __str__(self):
        return f"{self.exec_sec:.2f}s | {self.window} | {self.output}"


class StreamHelper:
    def __init__(self, s_context: int, t_poll: float, model: Speech2Text):
        self.s_context = s_context
        self.t_poll = t_poll
        self.model = model
        self.store: List[np.ndarray] = []
        self.golden = -1
        self.inference_task = None
        self.status: List[Status] = []

    def append_stream(self, stream: np.ndarray):
        self.store.append(stream)

        if self.inference_task and self.inference_task.done():
            self.inference_task = None

        if self.inference_task is None:
            self.inference_task = asyncio.create_task(
                asyncio.to_thread(self.exec_inference)  # 함수 명 다음으로 인자 줘도 됨
            )

    def get_stream(self):
        """return shallow copied stream"""
        len_str = len(self.store)
        start_index = max(self.golden + 1, len_str - self.s_context)
        return np.concatenate(self.store[start_index:]), (start_index, len_str - 1)

    def exec_inference(self):
        audio, window = self.get_stream()
        start_time = time.perf_counter()
        text = self.model(audio)[0][0]
        status = Status(window, text, time.perf_counter() - start_time)
        self.status.append(status)

    def get_status(self):
        return "\n".join(str(s) for s in self.status)

    def set_status(self, status: List[Status]):
        self.status = status
