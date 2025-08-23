import numpy as np
from typing import List, Tuple
import asyncio
import time
from espnet2.bin.asr_inference import Speech2Text
from dataclasses import dataclass
from realtime_joiner import RealtimeJoiner
# import yaml


# ex. [ ( (0, 3), "hello", 0.82) ] | ( (from, to), from_model, exe_time )[]
@dataclass
class Status:
    window: Tuple[int, int]
    output: str
    total: str
    exec_sec: float
    avg_db: float

    def __str__(self):
        return f"{self.exec_sec:.2f}s | {self.window} | {self.avg_db:.2f} | {self.output} > {self.total}"


class StreamHelper:
    def __init__(self, s_context: int, t_poll: float, model: Speech2Text):
        self.s_context = s_context
        self.t_poll = t_poll
        self.model = model
        self.store: List[np.ndarray] = []
        self.golden = -1
        self.inference_task = None
        self.status: List[Status] = []
        self.joiner = RealtimeJoiner()

        # with open(model_config['asr_train_config'], mode='r') as f:
        #     self.token_list: List[str] = yaml.safe_load(f)['token_list']
        # self.model.converter does it.

    def append_stream(self, stream: np.ndarray):
        self.store.append(stream)

        if self.inference_task and self.inference_task.done():
            self.inference_task = None

        if self.inference_task is None:
            self.inference_task = asyncio.create_task(
                asyncio.to_thread(self.exec_inference)  # 함수 명 다음으로 인자 줘도 됨
            )

    def get_stream(self):
        len_str = len(self.store)
        start_index = max(self.golden + 1, len_str - self.s_context)
        return np.concatenate(self.store[start_index:]), (start_index, len_str - 1)

    def get_average_decibels(self, audio_data: np.ndarray):
        """gen by gemini"""
        # 진폭을 제곱하여 전력을 계산하고, 평균을 구합니다.
        mean_square = np.mean(audio_data**2)

        # 0에 대한 로그를 방지하기 위해 작은 epsilon 값을 더합니다.
        epsilon = 1e-10

        # 데시벨로 변환합니다.
        average_decibels = 10 * np.log10(mean_square + epsilon)

        return average_decibels

    def exec_inference(self):
        audio, window = self.get_stream()
        db = self.get_average_decibels(audio)
        start_time = time.perf_counter()
        out = self.model(audio)
        text = out[0][0]

        # out[i] = (text, token, token_int, hyp)
        # print("=" * 10)
        # for i in range(len(out)):
        #     hyp = out[i][3]
        #     print(f"[{i}] total log probability: {hyp.score:.2f}")
        #     print(f"[{i}] normalized log probability: {hyp.score / len(hyp.yseq):.2f}")
        #     print(
        #         f"[{i}] hypo: "
        #         + "".join(self.model.converter.ids2tokens(hyp.yseq))
        #         + "\n"
        #     )
        # print("=" * 10)
        # text = "".join(self.model.converter.ids2tokens(out[0][3].yseq))

        self.status.append(
            Status(
                window,
                text,
                self.joiner.process_text(text, db),
                time.perf_counter() - start_time,
                db,
            )
        )

    def get_status(self):
        if not len(self.status):
            return "no-status"

        return "\n".join(str(s) for s in self.status[-10:])
