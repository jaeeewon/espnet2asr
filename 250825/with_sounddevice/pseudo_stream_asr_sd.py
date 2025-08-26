import sounddevice as sd
import numpy as np
import sys
from util.printer import printer
from dataclasses import dataclass
from typing import TypedDict, Tuple, Any
import threading
from infence_asr import Inference
from espnet2.bin.asr_inference import Speech2Text
from scipy.io.wavfile import write as write_wav


class Device(TypedDict):
    name: str
    index: int
    hostapi: int
    max_input_channels: int
    max_output_channels: int
    default_low_input_latency: float
    default_low_output_latency: float
    default_high_input_latency: float
    default_high_output_latency: float
    default_samplerate: float


class PseudoStreamASR_SD(Inference):
    def __init__(
        self,
        sr=16_000,
        channels=1,
        every=0.5,
        tag="Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best",
        device="cpu",
        context=6,
    ):
        super().__init__(
            s_context=context,
            t_poll=every,
            model=Speech2Text.from_pretrained(tag, device=device, nbest=1),
        )
        self.sr = sr
        self.channels = channels
        self.every = every

        self.device: Device = None
        self.stream: sd.InputStream = None

    def _clear_stream_gracefully(self):
        if self.stream is not None:
            if not self.stream.stopped:
                self.stream.stop()
            self.stream.close()
            self.stream = None

    def _select_devices(self):
        self._clear_stream_gracefully()

        devices = tuple(
            Device(d) for d in sd.query_devices() if d["max_input_channels"] > 0
        )
        assert len(devices), "at least one microphone is required to continue!"
        printer(
            (
                "automatically selected first microphone"
                if len(devices) > 1
                else "selected microphone device"
            ),
            devices[0],
        )
        self.device = devices[0]

        self.stream = sd.InputStream(
            samplerate=self.sr,
            device=self.device["index"],
            channels=self.channels,
            dtype=np.float32,
            blocksize=int(self.sr * self.every),
            callback=self._add_stream,
        )

    def _add_stream(
        self, indata: np.ndarray, frames: int, time: Any, status: sd.CallbackFlags
    ):
        with self.lock:
            self.audio.append(indata.copy())  # shallow copy | [:] not works, only .copy() works

        if self.inference_thread and not self.inference_thread.is_alive():
            self.inference_thread = None

        if self.inference_thread is None:
            self.inference_thread = threading.Thread(
                target=self._exec_inference, daemon=True
            )
            self.inference_thread.start()

    def start_stream(self):
        # assert self.device, "input device is not selected yet!"
        if self.device is None:
            self._select_devices()

        assert (
            self.stream.stopped
        ), "the stream you've just acceessed is already in use!"

        self.stream.start()
        print("started recording!")


asr = PseudoStreamASR_SD()
asr.start_stream()

# hold mainthread not being exitted
print("press enter or ctrl+c to exit")
input()
audio = np.concatenate(asr.audio[:])
write_wav("./test-audio.wav", asr.sr, audio)
