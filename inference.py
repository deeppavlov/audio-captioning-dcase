import numpy as np
import shutil
import os

from AudioCaption.captioning.pytorch_runners.inference_waveform import inference


def infer(audio_dir, model_path):
    def iteration(filename, filename_ix):
        print(f"Trying to infer on {filename}")
        cap = inference(
            os.path.join(audio_dir, filename),
            f"{filename_ix}.json", 
            model_path
        )
        return cap

    filenames = os.listdir(audio_dir)
    filenames = list(filter(lambda file: file.split(".")[-1] in ['wav', 'mp3', 'flac'], filenames))
    caps = []

    if isinstance(filenames, list):
        for filename_ix, filename in enumerate(filenames):
            caps.append(iteration(filename, filename_ix))
    else:
        caps.append(iteration(filenames, 0))
    print(f"caps: {caps}")


if __name__ == "__main__":
    audio_dir = '/home/ilia/projects/audio-captioning/AudioCaption/audiofiles'
    model_path = "/home/ilia/projects/audio-captioning/AudioCaption/experiments/clotho_v2/train_val/TransformerModel/cnn14rnn_trm/seed_1/swa.pth"
    infer(audio_dir, model_path)