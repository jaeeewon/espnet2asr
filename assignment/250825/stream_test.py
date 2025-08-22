# src: https://github.com/gradio-app/gradio/blob/main/demo/stream_audio/run.py
import gradio as gr
import numpy as np
import librosa
from espnet2.bin.asr_inference import Speech2Text

tag = "Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"
device = "cpu"

s2t = Speech2Text.from_pretrained(tag, device=device)

def add_to_stream(audio, audio_stream, text_stream):
    if audio is None:
        return gr.Audio(), text_stream, audio_stream, text_stream

    sr, y = audio
    y = y.astype(np.float32, order='C') / 32768.0
    y = librosa.resample(y, orig_sr=sr, target_sr=16000)

    if audio_stream is None:
        new_audio = audio
    else:
        new_audio = (sr, np.concatenate((audio_stream[1], y)))

    # text = f" [appended {y.dtype}] "
    text = s2t(new_audio[1])[0][0] + "\n"

    if text_stream is None or text_stream == "":
        new_text = text
    else:
        new_text = text_stream + text

    return new_audio, new_text, new_audio, new_text


with gr.Blocks() as demo:
    inp = gr.Audio(sources=["microphone"])
    out_audio = gr.Audio()
    out_text = gr.Textbox()
    audio_stream = gr.State()
    text_stream = gr.State()
    clear = gr.Button("Clear")

    inp.stream(
        add_to_stream,
        [inp, audio_stream, text_stream],
        [out_audio, out_text, audio_stream, text_stream],
    )
    clear.click(
        lambda: [None, None, "", None, None],
        None,
        [inp, out_audio, out_text, audio_stream, text_stream],
    )

if __name__ == "__main__":
    demo.launch()
