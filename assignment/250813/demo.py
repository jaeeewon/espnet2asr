from espnet2.bin.asr_inference import Speech2Text
# from espnet_model_zoo.downloader import ModelDownloader
import librosa
import gradio

tag = "Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"
device = "cuda"

# d = ModelDownloader()
# s2t = Speech2Text(
#     **d.download_and_unpack(tag),
#     device="cuda",
# )
s2t = Speech2Text.from_pretrained(tag, device=device)


def transcribe(path: str):
    raw, sr = librosa.load(path, sr=16000)
    # resampled = librosa.resample(y=raw, orig_sr=sr, target_sr=16000)

    y = s2t(raw)[0][0]
    return y

    # down-sampled가 더 잘 나타내더라
    # return f"{y_hat} (down-sampled: 16k) \n{'// SAME AS RAW' if y == y_hat else f'{y} (raw: {sr})'}"


demo = gradio.Interface(
    transcribe,
    gradio.Audio(
        sources="microphone",
        type="filepath"
    ),
    "text",
)

demo.launch()
