from asr_inference import Speech2Text
from util.joiner_abs import AbsJoiner
from util.joiner_v5 import JoinerV5 as Joiner
import soundfile
import time
import os
import numpy as np
import math
import threading
from jiwer import wer
from asr_align import CTCSegmentation
from espnet_model_zoo.downloader import ModelDownloader
from scipy.io.wavfile import write as write_wav
import shutil
from itertools import zip_longest
from util.printer import printer
from tqdm import tqdm
from glob import glob

special_token = {"sos_eos": 4999, "blank": 0, "unk": 1, "underscore": 4973}


class Comparison:
    def __init__(
        self,
        tag: str,
        device: str,
        sr: int,
        every: float,
        s_context: int,
        export_path="./segment",
    ):
        d = ModelDownloader()
        self.config = d.download_and_unpack(tag)
        self.model = Speech2Text(**self.config, device=device)
        self.segmentizer = CTCSegmentation(
            **self.config,
            fs=sr,
        )
        self.segmentizer.set_config(gratis_blank=True, kaldi_style_text=False)
        self.sr = sr
        self.every = every
        self.s_context = s_context
        self.under_inference = False
        self.lock = threading.Lock()
        self.export_path = export_path

        if os.path.exists(self.export_path):
            shutil.rmtree(self.export_path)

    def _load_wav(self, filename: str) -> np.ndarray[np.float32]:
        y, _sr = soundfile.read(filename, dtype="float32")
        return y
        # return np.concatenate(
        #     (y, np.zeros([int(self.sr * 2.5)]))
        # )  # 맨 마지막 부분이 context에서 사라질 때까지 데이터를 주고자

    def inference_origin(self, wav: np.ndarray) -> str:
        return self.model(wav)[0][0]

    def get_average_decibels(self, audio_data: np.ndarray):
        """gen by gemini"""
        # 진폭을 제곱하여 전력을 계산하고, 평균을 구합니다.
        mean_square = np.mean(audio_data**2)

        # 0에 대한 로그를 방지하기 위해 작은 epsilon 값을 더합니다.
        epsilon = 1e-10

        # 데시벨로 변환합니다.
        average_decibels = 10 * np.log10(mean_square + epsilon)

        return average_decibels

    def _thread_inference(
        self,
        wav: np.ndarray,
        joiner: AbsJoiner,
        filename: str,
        window: tuple[int, int],
    ):
        # w_start, w_end = window
        start_inf = time.perf_counter()
        out = self.model(wav)
        # out[i] = (text, token, token_int, hyp)
        # hyp = out[0][3]
        # print(f"model out {hyp.score:.2f} | {out[0][0]}")
        # print(hyp.states['lm'][0].shape)
        end_inf = time.perf_counter()
        # tokens = self.model.converter.ids2tokens(
        #     # out[0][3].yseq
        #     a
        #     for a in out[0][3].yseq
        #     if a not in (special_token["sos_eos"], special_token["underscore"])
        # )  # exclude sos/eos
        # tokens_str = "".join(tokens).replace(
        #     "▁", " "
        # )  # **important** ▁ is not underscore!!!!
        # tokens = tokens_str.split(" ")
        tokens = out[0][0].split(" ")
        # tokens = [
        #     f"{x} {y} {z}" for x, y, z in zip(tokens[0::3], tokens[1::3], tokens[2::3])
        # ]
        # tokens = out[0][0]  # just a str
        # tokens = "".join(tokens)
        # print("tokens", tokens)
        seg = self.segmentizer(wav, tokens)
        end_seg = time.perf_counter()
        # for index, start in enumerate(seg.segments):
        # print(
        #     f"{start:.2f} | {tokens[index] if len(tokens) > index else 'exceeds tokens'}"
        # )  # self.model.converter.token_list[id]}")
        # print(tokens)
        # print(seg)

        global_path = f"{self.export_path}/{filename}/{window}"
        global_file_path = f"{global_path}/{out[0][0]}.wav"
        if not os.path.exists(global_path):
            os.makedirs(global_path)

        # min_avg is confidence_score
        for i, (start, end, min_avg) in enumerate(seg.segments):
            # start end | confidence score | utterence ground truth
            # confidence score는 클 수록 좋음
            # print(f"{start:.2f} ~ {end:.2f} | {min_avg:3.4f} | {seg.text[i]}")
            wav_folderpath = f"{global_path}/{i}"
            wav_filename = f"{wav_folderpath}/{seg.text[i].replace('/', '_')}.wav"
            if not os.path.exists(wav_folderpath):
                os.makedirs(wav_folderpath)
            write_wav(
                wav_filename,
                self.sr,
                wav[math.floor(start * self.sr) : math.ceil(end * self.sr) + 1],
            )
            joiner.append_context(
                start_window=window[0],
                start_second=start,
                end_second=end,
                recognized=seg.text[i],
                min_avg=min_avg,
                segment_index=i,
            )
        # len_timings = len(seg.timings)
        # for (i, timing), token in zip(enumerate(seg.timings), tokens):
        #     start = timing
        #     end = (
        #         seg.timings[i + 1] if len_timings > i + 1 else seg.timings[-1]
        #     )  # next timing
        #     # start end | confidence score | utterence ground truth
        #     # confidence score는 클 수록 좋음
        #     print(f"{start:.2f} ~ {end:.2f} | {token}")
        #     wav_folderpath = f"{global_path}/{i}"
        #     wav_filename = f"{wav_folderpath}/{token.replace('/', '_')}.wav"
        #     if not os.path.exists(wav_folderpath):
        #         os.makedirs(wav_folderpath)
        #     write_wav(
        #         wav_filename,
        #         self.sr,
        #         wav[int(start * self.sr) : int(end * self.sr) + 1],
        #     )
        # print(
        #     f"[time elapsed] asr_inference: {(end_inf - start_inf):.2f}s | segmentize: {(end_seg - end_inf):.2f}s"
        # )
        # write_wav(global_file_path, self.sr, wav)
        # print(
        #     f"len timings: {len(seg.timings)} | len segments: {len(seg.segments)} | len yseq: {len(out[0][3].yseq)}"
        # )
        # print(seg)
        # print(global_file_path)
        # print("=" * 15)
        # joiner.process_text(out[0][0], self.get_average_decibels(wav)))

        with self.lock:
            self.under_inference = False

    def inference_pseudo_stream(
        self, wav: np.ndarray, filename="no-filename"
    ) -> list[str]:
        """
        0.5초마다 데이터를 불러와 최대 6개를 모델에 넣어주는 것처럼 동작해야 함
        """
        wav = np.concatenate(
            (wav, np.zeros([int(self.sr * self.every * (self.s_context - 1))]))
        )  # 맨 마지막 부분이 context에서 사라질 때까지 데이터를 주고자
        lenwav = len(wav)
        max_context = math.ceil(lenwav / (self.sr * self.every))
        start_time = time.perf_counter()
        last_time = start_time
        joiner: AbsJoiner = Joiner()
        until = 0
        while until < max_context:
            curr_time = time.perf_counter()

            if self.under_inference:
                time.sleep(0.01)
                continue

            if curr_time - last_time >= self.every:
                last_time = curr_time

                if not self.under_inference:
                    window_to = min(
                        math.floor((curr_time - start_time) / self.every) + 1,
                        max_context,
                    )

                    if window_to > until:
                        until = window_to

                        window_from = max(0, window_to - self.s_context + 1)
                        window_from_s = window_from * self.every
                        window_to_s = window_to * self.every

                        start = math.floor(window_from_s * self.sr)
                        end = math.ceil(window_to_s * self.sr)
                        target_wav = wav[start:end]

                        # print(
                        #     f"infer ({window_from}:{window_to}) | wav[{start}:{end}] out of {lenwav}"
                        # )

                        with self.lock:
                            self.under_inference = True

                        inf = threading.Thread(
                            target=self._thread_inference,
                            args=(
                                target_wav,
                                joiner,
                                filename,
                                (window_from, window_to),
                            ),
                        )
                        inf.start()

        while self.under_inference:
            time.sleep(0.01)

        return joiner.get_string()

    def execInference(self, dataset: str, path="../LibriSpeech/"):
        if dataset not in ("test-clean", "test-other"):
            raise AssertionError("invalid dataset " + dataset)

        root_dir = path + dataset
        sum_wer_answer_origin = 0
        sum_wer_origin_pseudo = 0
        sum_wer_answer_pseudo = 0
        count = 0

        for file in tqdm(glob(f"{root_dir}/**/*.trans.txt", recursive=True)):
            with open(file, encoding="utf-8") as f:
                for line in tqdm(f.readlines(), desc=file):
                    splt = line.strip().split(" ", maxsplit=1)
                    filename = f"{'/'.join(file.split('/')[:-1])}/{splt[0]}.flac"
                    answer = splt[1]
                    wav = self._load_wav(filename)
                    pseudo = self.inference_pseudo_stream(wav, f"{splt[0]}.flac")
                    origin = self.inference_origin(wav)
                    wer_answer_origin = wer(answer, origin)
                    wer_origin_pseudo = wer(origin, pseudo)
                    wer_answer_pseudo = wer(answer, pseudo)
                    print(
                        f"\n===== {filename} =====\nanswer: {answer}\norigin: {origin}\npseudo: {pseudo}\nWER(origin, pseudo): {wer_origin_pseudo:.2f}\nWER(answer, pseudo): {wer_answer_pseudo:.2f}"
                    )
                    sum_wer_answer_origin += wer_answer_origin
                    sum_wer_origin_pseudo += wer_origin_pseudo
                    sum_wer_answer_pseudo += wer_answer_pseudo
                    count += 1

                    printer(
                        f"pseudo-stream-asr-v5 with {self.every}s * {self.s_context} context using {dataset}",
                        {
                            "wer_answer_origin": sum_wer_answer_origin / count,
                            "wer_origin_pseudo": sum_wer_origin_pseudo / count,
                            "wer_answer_pseudo": sum_wer_answer_pseudo / count,
                        },
                    )

        # for i in os.listdir(root_dir):
        #     if i == ".DS_Store":
        #         continue
        #     path2 = f"{root_dir}/{i}"
        #     for ii in os.listdir(path2):
        #         if ii == ".DS_Store":
        #             continue
        #         path = f"{root_dir}/{i}/{ii}"
        #         with open(f"{path}/{i}-{ii}.trans.txt", encoding="utf-8") as f:
        #             for line in f:
        #                 splt = line.strip().split(" ", maxsplit=1)
        #                 filename = f"{path}/{splt[0]}.flac"
        #                 answer = splt[1]
        #                 wav = self._load_wav(filename)
        #                 pseudo = self.inference_pseudo_stream(
        #                     wav, f"{i}-{ii}-{splt[0]}.flac"
        #                 )
        #                 origin = self.inference_origin(wav)
        #                 wer_answer_origin = wer(answer, origin)
        #                 wer_origin_pseudo = wer(origin, pseudo)
        #                 wer_answer_pseudo = wer(answer, pseudo)
        #                 print(
        #                     f"\n===== {filename} =====\nanswer: {answer}\norigin: {origin}\npseudo: {pseudo}\nWER(origin, pseudo): {wer_origin_pseudo:.2f}\nWER(answer, pseudo): {wer_answer_pseudo:.2f}"
        #                 )
        #                 sum_wer_answer_origin += wer_answer_origin
        #                 sum_wer_origin_pseudo += wer_origin_pseudo
        #                 sum_wer_answer_pseudo += wer_answer_pseudo
        #                 count += 1

        printer(
            f"pseudo-stream-asr-v5 with {self.every}s * {self.s_context} context using {dataset}",
            {
                "wer_answer_origin": sum_wer_answer_origin / count,
                "wer_origin_pseudo": sum_wer_origin_pseudo / count,
                "wer_answer_pseudo": sum_wer_answer_pseudo / count,
            },
        )


comp = Comparison(
    sr=16_000,
    every=0.5,
    tag="Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best",
    device="cuda",
    s_context=6,
)
comp.execInference("test-clean")
