# src: https://github.com/gradio-app/gradio/blob/main/demo/stream_audio/run.py
import gradio as gr
import numpy as np

from espnet2.bin.asr_inference import Speech2Text
import librosa
import gradio

tag = "Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"
device = "cuda"

s2t = Speech2Text.from_pretrained(tag, device=device)


def add_to_stream(audio, instream):
    if audio is None:
        return gr.Audio(), instream
    if instream is None:
        ret = audio
    else:
        ret = (audio[0], np.concatenate((instream[1], audio[1])))
        print(ret[1].ndim, ret[1].shape)
    return ret, ret


with gr.Blocks() as demo:
    inp = gr.Audio(sources=["microphone"])
    out = gr.Audio()
    stream = gr.State()
    clear = gr.Button("Clear")

    inp.stream(
        add_to_stream,
        [inp, stream],
        [out, stream],
        stream_every=0.5,
    )
    clear.click(lambda: [None, None, None], None, [inp, out, stream])

if __name__ == "__main__":
    demo.launch()
