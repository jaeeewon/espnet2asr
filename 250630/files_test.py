import os

root_dir = "./LibriSpeech/test-clean"

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
                filename = splt[0] + ".flac"
                answer = splt[1]
                print({filename, answer})
