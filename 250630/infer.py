from espnet2.bin.asr_inference import Speech2Text
from wer import wer
import soundfile
import time
import os

"""
espnet // fork한 거 clone 후 pip install -e .
flash_attn
torchaudio
espnet_model_zoo
soundfile
"""

device = "cuda"  # "cuda" | "cpu"
tag = "Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"
# https://github.com/espnet/espnet_model_zoo/blob/master/espnet_model_zoo/table.csv
# 위에서 librispeech asr인 모델 아무거나 선택함

# Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best -> float32 -> flash attention 불가능
# 그래서 fork하여 flash attention 비활성화함
# failover가 없으니 성능 개선의 효과

s2t = Speech2Text.from_pretrained(
    tag,
    device=device,
)


def execInference(dataset: str):
    if dataset not in ("test-clean", "test-other"):
        raise AssertionError("invalid dataset " + dataset)

    root_dir = "./LibriSpeech/" + dataset
    sumWER = 0
    count = 0

    for i in os.listdir(root_dir):
        if i == ".DS_Store":
            continue
        for ii in os.listdir(f"{root_dir}/{i}"):
            if ii == ".DS_Store":
                continue
            path = f"{root_dir}/{i}/{ii}"
            with open(f"{path}/{i}-{ii}.trans.txt", encoding="utf-8") as f:
                for line in f:
                    splt = line.strip().split(" ", maxsplit=1)
                    filename = f"{path}/{splt[0]}.flac"
                    answer = splt[1]
                    speech = soundfile.read(filename)[
                        0
                    ]  # (speech, rate)에서 rate는 16k인 거 알고 있으니
                    t = time.perf_counter()
                    text = s2t(speech)[0][
                        0
                    ]  # s2t(speech)[0] -> text, token, token_int, hyp
                    w = wer(answer, text)
                    print(
                        f"{filename} elapsed {time.perf_counter() - t:.2f}s | {w:.2f} wer"
                    )
                    if w > 0.2:
                        print(f"inf: {text}\nans: {answer}")
                    sumWER += w
                    count += 1

    return sumWER / count

print('test-other WER', execInference('test-other'))

print('test-clean WER', execInference('test-clean'))