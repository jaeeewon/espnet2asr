from espnet2.bin.asr_inference import Speech2Text
import numpy as np
import librosa
import gradio

tag = "Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"
device = "cuda"

s2t = Speech2Text.from_pretrained(tag, device=device)


def transcribe(value: str | tuple[int, np.ndarray] | None):
    if value is None:
        return "no-input"

    if type(value) == str:
        raw, sr = librosa.load(value, sr=16000)
    elif type(value) == tuple:
        sr, raw = value
        # dtype of raw is int16 -> casting is required!

        # https://stackoverflow.com/a/42544738
        # it says, audio data with fp needs to be normalized between -1 and 1
        raw = raw.astype(np.float32, order='C') / 32768.0
        raw = librosa.resample(raw, orig_sr=sr, target_sr=16000)

    y = s2t(raw)[0][0]
    return y


demo = gradio.Interface(
    transcribe,
    gradio.Audio(sources="microphone", type="numpy"),
    "text",
)

demo.launch()
