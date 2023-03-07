import sys
import numpy as np
import pandas as pd
import shutil

sys.path.append("/src/aux_files")
sys.path.append("/src/aux_files/clotho-dataset")
sys.path.append("/src/aux_files/clotho-dataset/tools")
sys.path.append("/src/aux_files/wavetransformer")
sys.path.append("/src/aux_files/wavetransformer/wt_tools")

from sys import stdout
from pathlib import Path
from loguru import logger
from librosa.feature import melspectrogram
from tools.aux_functions import get_amount_of_file_in_dir, create_split_data
from wt_tools.file_io import load_numpy_object, dump_numpy_object
from typing import MutableMapping, MutableSequence,\
    Any, Union, List, Dict, Tuple
from torch import Tensor, no_grad
from torch.nn import Module, DataParallel
from wt_tools import file_io, printing
from wt_tools.model import module_epoch_passing, get_model, get_device
from data_handlers.clotho_loader import get_clotho_loader


def prepare_dataset(settings_dataset, settings_features, config, inference_params):

    logger.remove()
    logger.add(stdout, format='{level} | [{time:HH:mm:ss}] {name} -- {message}.',
                level='INFO', filter=lambda record: record['extra']['indent'] == 1)
    logger.add(stdout, format='  {level} | [{time:HH:mm:ss}] {name} -- {message}.',
                level='INFO', filter=lambda record: record['extra']['indent'] == 2)
    main_logger = logger.bind(indent=1)
    
    settings_dataset = {
        "verbose": True,
        "directories": {
            "root_dir": inference_params['dataset_rootdir'], #/content/data
            "annotations_dir": '',
            "downloaded_audio_dir": 'clotho_audio_files',
        },
        "output_files": {
            "dir_output": 'data_splits',
            "file_name_template": 'clotho_file_{audio_file_name}.npy',
        },
        "audio": {
            "sr": 44100,
            "to_mono": True,
            "max_abs_value": 1.,
        },
        "counters": {
            "words_list_file_name": 'WT_pickles/WT_words_list.p',
            "words_counter_file_name": 'WT_pickles/words_frequencies.p',
            "characters_list_file_name": 'WT_pickles/WT_characters_list.p',
            "characters_frequencies_file_name": 'WT_pickles/characters_frequencies.p',
        }
    }


    settings_features = {
        "package": 'processes',
        "module": 'features_log_mel_bands',
        "data_files_suffix": '.npy',
        "keep_raw_audio_data": False,
        "output": {
            "dir_output": inference_params['features_output_dir'], #/content/data/clotho_dataset
        },
        "process": {
            "sr": 44100,
            "nb_fft": 1024,
            "hop_size": 512,
            "nb_mels": 64,
            "window_function": 'hann',
            "center": True,
            "f_min": .0,
            "f_max": None,
            "htk": False,
            "power": 1.,
            "norm": 1,
        }
    }


    config = {
        "dnn_training_settings": {
            "model": {
                "model_name": 'wave_transformer_3',
                "encoder": {
                    "in_channels_encoder":  64,
                    "out_channels_encoder": [16,32,64,128],
                    "kernel_size_encoder": 3,
                    "dilation_rates_encoder": [2,2,2,2],
                    "last_dim_encoder": 128,
                    "beam_size": 0,
                },
                "decoder": {
                    "num_layers_decoder": 3,
                    "num_heads_decoder": 4,
                    "n_features_decoder": 128,
                    "n_hidden_decoder": 128,
                    "nb_classes": 4367,
                    "dropout_decoder": .25,
                }
            },
            "data": {
                "input_field_name": "features",
                'output_field_name': 'words_ind',
                "load_into_memory": False,
                "batch_size": 1,
                "num_workers": 30,
                "use_multiple_mode": False
            },
        },
        "dirs_and_files": {
            "root_dirs": {
                "outputs": inference_params['pretrained_models_dir'], #'/content/audio-captioning-dcase/wavetransformer/outputs',
                "model_data": inference_params['pretrained_pickle_files_dir'], #'/content/audio-captioning-dcase/wavetransformer/data',
                "data": inference_params['dataset_rootdir'],
            },
            "dataset": {
                "features_dirs": {
                    "output": 'data_splits_features',
                },
                "audio_dirs": {
                    "downloaded": 'clotho_audio_files',
                    "output": 'data_splits_audio',
                },
                "pickle_files_dir": 'WT_pickles',
                "files": {
                    "np_file_name_template": 'clotho_file_{audio_file_name}_{caption_index}.npy',
                    "words_list_file_name": 'WT_words_list.p',
                    "characters_list_file_name": 'WT_character_list.p'
                }
            },
            "model": {
                "model_dir": 'models',
                "pre_trained_model_name": 'best_model_16_3_9.pt'
            },
            "logging": {
                "logger_dir": 'logging',
                "caption_logger_file": 'caption.txt'
            }
        }
    }

    if not settings_dataset['verbose']:
        main_logger.info('Verbose if off. Not logging messages')
        logger.disable('__main__')
        logger.disable('processes')

    main_logger.info('Starting dataset creation')

    dir_root = Path(settings_dataset['directories']['root_dir'])
    dir_out = dir_root.joinpath(settings_dataset['output_files']['dir_output'])
    dir_audio = Path(settings_dataset['directories']['downloaded_audio_dir'])

    main_logger.info('Creating the .npy files')
    create_split_data(dir_out, dir_audio, dir_root, 
                      settings_dataset['audio'], settings_dataset['output_files'])
    main_logger.info('Done')

    nb_files_audio = get_amount_of_file_in_dir(dir_root.joinpath(dir_audio))
    nb_files_data = get_amount_of_file_in_dir(dir_out)

    main_logger.info('Amount of audio files: {}'.format(nb_files_audio))
    main_logger.info('Amount of data files: {}'.format(nb_files_data))
    main_logger.info('Amount of data files per audio: {}'.format(nb_files_data / nb_files_audio))

    main_logger.info('Done')

    main_logger.info('Dataset created')

    dir_output = dir_root.joinpath(dir_out)

    for data_file_name in filter(lambda _x: _x.suffix == settings_features['data_files_suffix'],
                                dir_output.iterdir()):

        data_file = load_numpy_object(data_file_name)

        features = feature_extraction(data_file['audio_data'].item(),
                        **settings_features['process'])

        array_data = (data_file['file_name'].item(), )
        dtypes = [('file_name', data_file['file_name'].dtype)]

        if settings_features['keep_raw_audio_data']:
            array_data += (data_file['audio_data'].item(), )
            dtypes.append(('audio_data', data_file['audio_data'].dtype))

        array_data += (features, data_file['words_ind'].item(),)
        dtypes.extend([('features', np.ndarray), ('words_ind', data_file['words_ind'].dtype)])

        main_logger.info("adata with features: {}", array_data)

        np_rec_array = np.rec.array([array_data], dtype=dtypes)

        file_path = Path(settings_features['output']['dir_output']).joinpath(data_file_name.name)

        dump_numpy_object(np_rec_array, str(file_path)) # save to var, not to file


    main_logger.info('Features extracted')

    shutil.move("/src/data/data_splits_features", "/src/data/data_splits_features_OLD")
    shutil.move("/src/data/clotho_dataset", "/src/data/data_splits_features")

    return settings_dataset, settings_features, config


def feature_extraction(audio_data: np.ndarray, sr: int, nb_fft: int,
                    hop_size: int, nb_mels: int, f_min: float,
                    f_max: float, htk: bool, power: float, norm: bool,
                    window_function: str, center: bool) -> np.ndarray:
    """Feature extraction function.

    :param audio_data: Audio signal.
    :type audio_data: numpy.ndarray
    :param sr: Sampling frequency.
    :type sr: int
    :param nb_fft: Amount of FFT points.
    :type nb_fft: int
    :param hop_size: Hop size in samples.
    :type hop_size: int
    :param nb_mels: Amount of MEL bands.
    :type nb_mels: int
    :param f_min: Minimum frequency in Hertz for MEL band calculation.
    :type f_min: float
    :param f_max: Maximum frequency in Hertz for MEL band calculation.
    :type f_max: float|None
    :param htk: Use the HTK Toolbox formula instead of Auditory toolkit.
    :type htk: bool
    :param power: Power of the magnitude.
    :type power: float
    :param norm: Area normalization of MEL filters.
    :type norm: bool
    :param window_function: Window function.
    :type window_function: str
    :param center: Center the frame for FFT.
    :type center: bool
    :return: Log mel-bands energies of shape=(t, nb_mels)
    :rtype: numpy.ndarray
    """
    y = audio_data/abs(audio_data).max()
    mel_bands = melspectrogram(
        y=y, sr=sr, n_fft=nb_fft, hop_length=hop_size, win_length=nb_fft,
        window=window_function, center=center, power=power, n_mels=nb_mels,
        fmin=f_min, fmax=f_max, htk=htk, norm=norm).T

    return np.log(mel_bands + np.finfo(float).eps)


class MyDataParallel(DataParallel):

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def _decode_outputs(predicted_outputs: MutableSequence[Tensor],
                    model_indices_object: MutableSequence[str],
                    file_names: MutableSequence[Path],
                    eos_token: str,
                    print_to_console: bool) \
        -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    caption_logger = logger.bind(is_caption=True, indent=None)
    main_logger = logger.bind(is_caption=False, indent=0)
    caption_logger.info('Captions start')
    main_logger.info('Starting decoding of captions')

    captions_pred: List[Dict] = []
    f_names: List[str] = []

    for b_predictions, f_name in zip(
            predicted_outputs, file_names):
        print(b_predictions.float(), "\n\n", b_predictions.cpu().numpy())
        predicted_words = b_predictions
        predicted_caption = [model_indices_object[i.item()]
                            for i in predicted_words]
        
        try:
            predicted_caption = predicted_caption[
                :predicted_caption.index(eos_token)]
        except ValueError:
            pass

        predicted_caption = ' '.join(predicted_caption)

        f_n = f_name.stem

        if f_n not in f_names:
            f_names.append(f_n)
            captions_pred.append({
                'file_name': f_n,
                'caption_predicted': predicted_caption})

        log_strings = [f'Captions for file {f_name.stem}: ',
                    f'\tPredicted caption: {predicted_caption}']

        [caption_logger.info(log_string)
        for log_string in log_strings]

        if print_to_console:
            [main_logger.info(log_string)
            for log_string in log_strings]

    logger.bind(is_caption=False, indent=0).info(
        'Decoding of captions ended')

    return captions_pred

def _do_inference(model: Module,
                settings_data:  MutableMapping[str, Any],
                settings_io:  MutableMapping[str, Any],
                model_indices_list: MutableSequence[str],
                caps) \
        -> None:
    model.eval()
    logger_main = logger.bind(is_caption=False, indent=1)

    logger_main.info('Getting inference data')
    inference_data = get_clotho_loader(settings_data=settings_data, settings_io=settings_io)

    logger_main.info('Done')

    with no_grad():
        outputs = module_epoch_passing(
            data=inference_data, module=model,
            use_y=False,
            objective=None, optimizer=None)
    captions_pred = _decode_outputs(
        outputs[1],
        model_indices_object=model_indices_list,
        file_names=outputs[3],
        eos_token='<eos>',
        print_to_console=False)

    logger_main.info('Inference done')

    caps = pd.DataFrame(captions_pred)
    logger_main.info("caps: {}", caps)
    return caps
    

def _load_indices_file(settings_files: MutableMapping[str, Any]) \
        -> MutableSequence[str]:
    path = Path(
        settings_files['root_dirs']['model_data'],
        settings_files['dataset']['pickle_files_dir'])
    print(settings_files['root_dirs']['model_data'])
    print(settings_files['dataset']['pickle_files_dir'])
    indices_filename = path.joinpath(settings_files['root_dirs']['data'],
                                        settings_files['dataset']['pickle_files_dir'],
                                        settings_files['dataset']['files']['words_list_file_name'])
    
    return file_io.load_pickle_file(indices_filename)


def method(settings: MutableMapping[str, Any], caps) -> None:
    logger_main = logger.bind(is_caption=False, indent=0)
    logger_main.info('Bootstrapping method')
    pretty_printer = printing.get_pretty_printer()
    logger_inner = logger.bind(is_caption=False, indent=1)

    device, device_name = get_device(0)                     # IF NO GPU, IT'LL FAIL

    model_dir = Path(
        settings['dirs_and_files']['root_dirs']['outputs'],
        settings['dirs_and_files']['model']['model_dir']
    )

    model_dir.mkdir(parents=True, exist_ok=True)

    logger_inner.info(f'Process on {device_name}\n')

    logger_inner.info('Settings:\n'
                    f'{pretty_printer.pformat(settings)}\n')
    
    model_indices_list = _load_indices_file(settings['dirs_and_files'])
    logger_inner.info('Done')

    model: Union[Module, None] = None

    logger_main.info('Bootstrapping done')

    logger_main.info('Loading model')
    if model is None:
        logger_inner.info('Setting up model')
        model = get_model(
            settings_model=settings['dnn_training_settings']['model'],
            settings_io=settings['dirs_and_files'],
            output_classes=settings['dnn_training_settings']['model']['decoder']['nb_classes'],
            device=device)
        model.to(device)
        logger_inner.info('Model ready')
    logger_inner.info('Starting inference')
    caps = _do_inference(
        caps=caps,
        model=model,
        settings_data=settings['dnn_training_settings']['data'],
        settings_io=settings['dirs_and_files'],
        model_indices_list=model_indices_list)
    logger_inner.info('Inference done')
    return caps

def infer(inference_params):
    caps = settings_dataset = settings_features = config = None
    settings_dataset, settings_features, config = \
        prepare_dataset(settings_dataset, 
                        settings_features, config, inference_params)
    caps = method(config)

    return caps