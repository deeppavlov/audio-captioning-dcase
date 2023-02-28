#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import count
from collections import deque
from pathlib import Path
from typing import MutableMapping, Any

import numpy as np
import os

from tools.file_io import load_audio_file, dump_numpy_object

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['get_amount_of_file_in_dir', 'get_annotations_files',
           'check_data_for_split', 'create_split_data',
           'create_lists_and_frequencies']


def get_amount_of_file_in_dir(the_dir: Path) -> int:
    """Counts the amount of files in a directory.

    :param the_dir: Directory.
    :type the_dir: pathlib.Path
    :return: Amount of files in directory.
    :rtype: int
    """
    counter = count()

    deque(zip(the_dir.iterdir(), counter))

    return next(counter)


def create_split_data(dir_split: Path,
                      dir_audio: Path, dir_root: Path,
                      settings_audio: MutableMapping[str, Any],
                      settings_output: MutableMapping[str, Any]) -> None:
    """Creates the data for the split.

    :param dir_split: Directory for the split.
    :type dir_split: pathlib.Path
    :param dir_audio: Directory of the audio files for the split.
    :type dir_audio: pathlib.Path
    :param dir_root: Root directory of data.
    :type dir_root: pathlib.Path
    :param settings_audio: Settings for the audio.
    :type settings_audio: dict
    :param settings_output: Settings for the output files.
    :type settings_output: dict
    """

    dir_split.mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(os.path.join(dir_root, dir_audio)):

        if not os.path.isfile(os.path.join(dir_root, dir_audio, filename)) \
            or filename[-4:] != ".wav": # TOFIX: VERY HARDCODED, BAD BAD BAD
            continue

        audio = load_audio_file(
            audio_file=str(os.path.join(dir_root, dir_audio, filename)),
            sr=int(settings_audio['sr']), mono=settings_audio['to_mono'])

        np_rec_array = np.rec.array(np.array(
            (filename, audio),
            dtype=[
                ('file_name', 'U{}'.format(len(filename))),
                ('audio_data', np.dtype(object))
            ]
        ))

        dump_numpy_object(
            np_obj=np_rec_array,
            file_name=str(dir_split.joinpath(
                settings_output['file_name_template'].format(
                    audio_file_name=filename))))

# EOF
