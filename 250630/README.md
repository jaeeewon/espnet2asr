# prerequisites
download and extract into ./LibriSpeech so that it contains './LibriSpeech/test-<clean | other>/'\
https://www.openslr.org/resources/12/test-clean.tar.gz

https://www.openslr.org/resources/12/test-other.tar.gz

# requirements
```
git clone https://github.com/jaeeewon/espnet.git
cd espnet
pip install -e .
```
`pip install jiwer torchaudio espnet_model_zoo soundfile`

# entry point
`<python> infer.pyp`