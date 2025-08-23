# src: https://github.com/gradio-app/gradio/blob/main/demo/stream_audio/run.py
import gradio as gr
import numpy as np
import librosa
from stream_helper import StreamHelper
from espnet2.bin.asr_inference import Speech2Text

"""
# TODO
1. 가장 최근 출력과 context개 출력의 CER/WER 비교하여 일정 이상이면 이를 정답으로 잠정 결론
2. 그 이전의 음성은 모델에 입력하지 않고, 이후부터 입력함
3. 토큰이 every초 사이에 잘릴 수 있으니 context와 every를 적절히 조정해야

** 아이디어는 아래에서 나옴 **
1.10s | AND
1.72s | HELLO HOWITZER I WAS WAITING
2.07s | HELLO HOWITZER I WAS WAITING FOR YOU
2.38s | HELLO HOW WAS YOUR DAY I WAS WAITING FOR YOU
출력이 더 많이 있으면 언어 모델이 더 잘 표현해주기 때문에, 일정 이상 보정되면 이를 정답으로 보자
"""


class PseudoStreamASR:
    def __init__(
        self,
        tag="Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best",
        device="cpu",
        context=6,
        every=0.5,
        sr=16_000,
        threshold=0.7,
        autorun=True,
    ):
        self.every = every
        self.sr = sr
        self.helper = StreamHelper(
            s_context=context,
            t_poll=every,
            model=Speech2Text.from_pretrained(tag, device=device),
        )
        if autorun:
            self.__call__()

    def __call__(self):
        with gr.Blocks() as demo:
            inp_audio = gr.Audio(sources=["microphone"])
            out_audio = gr.Audio()

            out_text = gr.Textbox()
            audio_stream = gr.State()
            text_stream = gr.State()
            clear = gr.Button("Clear")

            inp_audio.stream(
                fn=self.append_stream,
                inputs=[inp_audio, audio_stream, text_stream],
                outputs=[out_audio, out_text, audio_stream, text_stream],
                stream_every=self.every,
            )

            clear.click(
                lambda: [None, None, "", None, None],
                None,
                [inp_audio, out_audio, out_text, audio_stream, text_stream],
            )
        demo.launch()

    async def append_stream(self, audio, audio_stream, text_stream):
        if audio is None:
            return gr.Audio(), text_stream, audio_stream, text_stream

        _sr, y = audio
        y = y.astype(np.float32, order="C") / 32768.0
        y = librosa.resample(y, orig_sr=_sr, target_sr=self.sr)

        if audio_stream is None:
            new_audio = (self.sr, y)
        else:
            new_audio = (self.sr, np.concatenate((audio_stream[1], y)))

        self.helper.append_stream(y)
        text = self.helper.get_status()

        return new_audio, text, new_audio, text


if __name__ == "__main__":
    PseudoStreamASR(context=6, every=1)
