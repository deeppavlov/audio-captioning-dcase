import numpy as np
import shutil
import os
import sys

sys.path.append("/src/aux_files/AudioCaption/captioning")
sys.path.append("/src/aux_files/AudioCaption")
sys.path.append("/src/aux_files")

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
    if len(filenames) == 1:
        print("INFERENCE LOG - inferring on one file")
        [filenames] = filenames
    else:
        print("INFERENCE LOG - inferring multiple files at a time")
    caps = []

    if isinstance(filenames, list):
        for filename_ix, filename in enumerate(filenames):
            caps.append(iteration(filename, filename_ix))
        print(f"INFERENCE LOG - caps: {caps}")
        return caps
    else:
        caps = iteration(filenames, 0)
        print(f"INFERENCE LOG - cap: {caps}")
        return caps