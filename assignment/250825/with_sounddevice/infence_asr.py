import numpy as np
from espnet2.bin.asr_inference import Speech2Text
from typing import Tuple, List
from dataclasses import dataclass
from util.realtime_joiner import RealtimeJoiner
import time
import threading


@dataclass
class Context:
    window: Tuple[int, int]
    output: str = None
    total: str = None
    exec_sec: float = None
    avg_db: float = None

    def __str__(self):
        return f"{self.exec_sec:.2f}s | {self.window} | {self.avg_db:.2f} | {self.output} > {self.total}"


class Inference:
    def __init__(self, s_context: int, t_poll: float, model: Speech2Text):
        self.s_context = s_context
        self.t_poll = t_poll
        self.model = model

        self.audio: list[np.ndarray] = []
        self.inference_thread: threading.Thread = None
        self.golden = -1
        self.context: List[Context] = []
        self.joiner = RealtimeJoiner()
        self.lock = threading.Lock()

        self.printed = False

    def get_stream(self):
        with self.lock:
            len_str = len(self.audio)
            start_index = max(self.golden + 1, len_str - self.s_context)
            return np.concatenate(self.audio[start_index:]), (start_index, len_str - 1)

    def get_average_decibels(self, audio_data: np.ndarray):
        """gen by gemini"""
        # 진폭을 제곱하여 전력을 계산하고, 평균을 구합니다.
        mean_square = np.mean(audio_data**2)

        # 0에 대한 로그를 방지하기 위해 작은 epsilon 값을 더합니다.
        epsilon = 1e-10

        # 데시벨로 변환합니다.
        average_decibels = 10 * np.log10(mean_square + epsilon)

        return average_decibels

    def _exec_inference(self):
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

        self.context.append(
            Context(
                window,
                text,
                self.joiner.process_text(text, db),
                time.perf_counter() - start_time,
                db,
            )
        )
        self.refresh_status()

    def get_status(self):
        if not len(self.context):
            return "no-status"

        return "\n".join(str(s) for s in self.context[-10:])

    def refresh_status(self):
        # print("\033[H\033[2J") # clear screen
        if self.printed:
            # n줄 삭제 및 다시
            print("\033[3A", end="")

        print(f"\r\033[K실시간 대시보드")
        print(f"\r\033[K[recognized] {self.context[-1].output}")
        print(f"\r\033[K[in totally] {self.context[-1].total}")
        self.printed = True
