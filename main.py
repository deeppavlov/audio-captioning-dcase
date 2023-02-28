    # EMULATING DOCKERFILE

    %cd /content/

    !echo "--------------------------------removing old files...--------------------------------"
    !rm -rf /content/sample_data
    !rm -rf /content/audio-captioning-dcase
    !rm -rf /content/data

    !echo "--------------------------------cloning all required repositories...--------------------------------"
    !git clone https://github.com/moon-strider/audio-captioning-dcase

    !echo "--------------------------------installing dependencies...--------------------------------"
    !pip install -r /content/audio-captioning-dcase/wavetransformer/requirement_pip.txt

    !echo "--------------------------------copying partial dataset from google drive...--------------------------------"
    !cp -r /content/drive/MyDrive/dcasePart/evaluation /content/audio-captioning-dcase/wavetransformer/data/clotho_audio_files/
    !cp -r /content/drive/MyDrive/dcasePart/development /content/audio-captioning-dcase/wavetransformer/data/clotho_audio_files/

    #!cp /content/drive/MyDrive/dcasePart/clotho_captions_development.csv /content/audio-captioning-dcase/wavetransformer/data/clotho_csv_files/clotho_captions_development.csv
    #!cp /content/drive/MyDrive/dcasePart/clotho_captions_evaluation.csv /content/audio-captioning-dcase/wavetransformer/data/clotho_csv_files/clotho_captions_evaluation.csv

    !echo "--------------------------------copying development -> evaluation...--------------------------------"
    !cp -r /content/audio-captioning-dcase/wavetransformer/data /content/audio-captioning-dcase/clotho-dataset
    #!cp /content/audio-captioning-dcase/clotho-dataset/data/clotho_csv_files/clotho_captions_development.csv /content/audio-captioning-dcase/clotho-dataset/data/clotho_csv_files/clotho_captions_evaluation.csv
    !cp -r /content/audio-captioning-dcase/clotho-dataset/data/clotho_audio_files/development/* /content/audio-captioning-dcase/clotho-dataset/data/clotho_audio_files/evaluation/

    !cp -r /content/audio-captioning-dcase/clotho-dataset/data /content/
    !rm -f /content/data/data_splits_features/evaluation/.dummy
    !rm -f /content/data/data_splits_features/development/.dummy

    # EMULATING DOCKERFILE


    import sys
    import numpy as np
    import pandas as pd
    import copy
    import shutil

    # sys.path.append("./aux_files/wavetransformer")
    # sys.path.append("./aux_files/wavetransformer/wt_tools")
    # sys.path.append("./aux_files/clotho-dataset/tools")
    # sys.path.append("./aux_files/clotho-dataset")

    sys.path.append("/content/audio-captioning-dcase/wavetransformer")
    sys.path.append("/content/audio-captioning-dcase/wavetransformer/wt_tools")
    sys.path.append("/content/audio-captioning-dcase/clotho-dataset/tools")
    sys.path.append("/content/audio-captioning-dcase/clotho-dataset")

    from sys import stdout
    from pathlib import Path
    from loguru import logger
    from librosa.feature import melspectrogram
    from aux_functions import get_amount_of_file_in_dir, create_split_data
    from wt_tools.file_io import load_numpy_object, dump_numpy_object
    from itertools import chain
    from typing import MutableMapping, MutableSequence,\
        Any, Union, List, Dict, Tuple
    from torch import Tensor, no_grad
    from torch.nn import Module, DataParallel
    from wt_tools import file_io, printing
    from wt_tools.model import module_epoch_passing, get_model, get_device
    from data_handlers.clotho_loader import get_clotho_loader


    caps = settings_dataset = settings_features = config = features_drop = None


    def prepare_dataset():
        global settings_dataset
        global settings_features
        global config
        global features_drop

        logger.remove()
        logger.add(stdout, format='{level} | [{time:HH:mm:ss}] {name} -- {message}.',
                    level='INFO', filter=lambda record: record['extra']['indent'] == 1)
        logger.add(stdout, format='  {level} | [{time:HH:mm:ss}] {name} -- {message}.',
                    level='INFO', filter=lambda record: record['extra']['indent'] == 2)
        main_logger = logger.bind(indent=1)

        # placeholder for loading the data maybe
        
        settings_dataset = {
            "verbose": True,
            "directories": {
                "root_dir": '/content/data',
                "annotations_dir": '',
                "downloaded_audio_dir": 'clotho_audio_files',
                "downloaded_audio_development": 'development',
                "downloaded_audio_evaluation": 'evaluation',
            },
            "output_files": {
                "dir_output": 'data_splits',
                "dir_data_development": 'development',
                "dir_data_evaluation": 'evaluation',
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
                "dir_output": '/content/data',
                "dir_development": 'clotho_dataset_dev',
                "dir_evaluation": 'clotho_dataset_eva',
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
                    "use_validation_split": False,
                    "input_field_name": "features",
                    "output_field_name": "words_ind",
                    "load_into_memory": False,
                    "batch_size": 1,
                    "shuffle": True,
                    "num_workers": 30,
                    "drop_last": True,
                    "use_multiple_mode": False
                },
            },
            "dirs_and_files": {
                "root_dirs": {
                    "outputs": '/content/audio-captioning-dcase/wavetransformer/outputs',
                    "model_data": '/content/audio-captioning-dcase/wavetransformer/data',
                    "data": '/content/data',
                },
                "dataset": {
                    "development": "development",
                    "evaluation": "evaluation",
                    "validation": "validation",
                    "features_dirs": {
                        "output": 'data_splits_features',
                        "development": "development",
                        "evaluation": "evaluation",
                        "validation": "validation",
                    },
                    "audio_dirs": {
                        "downloaded": 'clotho_audio_files',
                        "output": 'data_splits_audio',
                        "development": "development",
                        "evaluation": "evaluation",
                        "validation": "validation",
                    },
                    "pickle_files_dir": 'WT_pickles',
                    "files": {
                        "np_file_name_template": 'clotho_file_{audio_file_name}_{caption_index}.npy',
                        "words_list_file_name": 'WT_words_list.p',
                        "characters_list_file_name": 'WT_characters_list.p'
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

        main_logger.info('Starting Clotho dataset creation')

        dir_root = Path(settings_dataset['directories']['root_dir'])

        # TODO: DEV, EVA, VAL everywhere to INPUT
        for split_name in ['development', 'evaluation']:

            dir_split = dir_root.joinpath(
            settings_dataset['output_files']['dir_output'],
            settings_dataset['output_files']['dir_data_{}'.format(split_name)])

            dir_downloaded_audio = Path(
            settings_dataset['directories']['downloaded_audio_dir'],
            settings_dataset['directories']['downloaded_audio_{}'.format(split_name)])

            main_logger.info("dir_split: {}", dir_split)
            main_logger.info("dir_audio: {}", dir_downloaded_audio)
            main_logger.info("dir_root: {}", dir_root)

            main_logger.info('Creating the {} split data'.format(split_name))
            create_split_data(dir_split, dir_downloaded_audio, dir_root, 
                              settings_dataset['audio'], settings_dataset['output_files'])
            main_logger.info('Done')

            nb_files_audio = get_amount_of_file_in_dir(
            dir_root.joinpath(dir_downloaded_audio))
            nb_files_data = get_amount_of_file_in_dir(dir_split)

            main_logger.info('Amount of {} audio files: {}'.format(
            split_name, nb_files_audio))
            main_logger.info('Amount of {} data files: {}'.format(
            split_name, nb_files_data))
            main_logger.info('Amount of {} data files per audio: {}'.format(
            split_name, nb_files_data / nb_files_audio))

            main_logger.info('Done')

        main_logger.info('Dataset created')

        dir_output = dir_root.joinpath(settings_dataset['output_files']['dir_output'])
        dir_dev = dir_output.joinpath(
            settings_dataset['output_files']['dir_data_development'])
        dir_eva = dir_output.joinpath(
            settings_dataset['output_files']['dir_data_evaluation'])

        dir_output_dev = dir_root.joinpath(
                settings_features['output']['dir_output'],
                settings_features['output']['dir_development'])
        dir_output_eva = dir_root.joinpath(
                settings_features['output']['dir_output'],
                settings_features['output']['dir_evaluation'])

        dir_output_dev.mkdir(parents=True, exist_ok=True)
        dir_output_eva.mkdir(parents=True, exist_ok=True)

        for data_file_name in filter(lambda _x: _x.suffix == settings_features['data_files_suffix'],
                                    chain(dir_dev.iterdir(), dir_eva.iterdir())):

            data_file = load_numpy_object(data_file_name)

            features = feature_extraction(data_file['audio_data'].item(),
                            **settings_features['process'])

            array_data = (data_file['file_name'].item(), )
            dtypes = [('file_name', data_file['file_name'].dtype)]

            if settings_features['keep_raw_audio_data']:
                array_data += (data_file['audio_data'].item(), )
                dtypes.append(('audio_data', data_file['audio_data'].dtype))

            array_data += (features)
            dtypes.extend([('features', np.dtype(object))])

            main_logger.info("adata with features: {}", array_data)

            np_rec_array = np.rec.array([array_data], dtype=dtypes)

            parent_path = dir_output_dev \
                if data_file_name.parent.name == settings_dataset['output_files']['dir_data_development'] \
                else dir_output_eva

            file_path = parent_path.joinpath(data_file_name.name)

            dump_numpy_object(np_rec_array, str(file_path)) # save to var, not to file
        #features_drop = np_rec_array


        main_logger.info('Features extracted')

        shutil.move("/content/data/data_splits_features/development", "/content/data/data_splits_features/devOLD")
        shutil.move("/content/data/data_splits_features/evaluation", "/content/data/data_splits_features/evalOLD")
        
        shutil.move("/content/data/clotho_dataset_dev", "/content/data/data_splits_features/development")
        shutil.move("/content/data/clotho_dataset_eva", "/content/data/data_splits_features/evaluation")
        
        #subprocess.run(["echo", "\'--------------------------------dataset -> data_splits...--------------------------------\'"])
        #subprocess.run(["cp", "-r", "/content/audio-captioning-dcase/clotho-dataset/data/clotho_dataset_dev/*", "/content/audio-captioning-dcase/clotho-dataset/data/data_splits_features/development"])
        #subprocess.run(["cp", "-r", "/content/audio-captioning-dcase/clotho-dataset/data/clotho_dataset_eva/*", "/content/audio-captioning-dcase/clotho-dataset/data/data_splits_features/evaluation"])

        #subprocess.run(["cp", "-rf", "/content/audio-captioning-dcase/wavetransformer/data/WT_pickles", "/content/audio-captioning-dcase/clotho-dataset/data/WT_pickles"])

        #subprocess.run(["cp", "-r", "/content/audio-captioning-dcase/clotho-dataset/data/data_splits_features/evaluation", "/content/audio-captioning-dcase/clotho-dataset/data/data_splits_features/validation"])
        #subprocess.run(["cp", "-r", "/content/audio-captioning-dcase/clotho-dataset/data/clotho_audio_files/evaluation", "/content/audio-captioning-dcase/clotho-dataset/data/clotho_audio_files/validation"])
        #subprocess.run(["cp", "/content/audio-captioning-dcase/clotho-dataset/data/clotho_csv_files/clotho_captions_evaluation.csv", "/content/audio-captioning-dcase/clotho-dataset/data/clotho_csv_files/clotho_captions_validation.csv"])


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
                    model_indices_list: MutableSequence[str]) \
            -> None:
        global caps
        model.eval()
        logger_main = logger.bind(is_caption=False, indent=1)

        logger_main.info('Getting inference data')
        validation_data = get_clotho_loader(
            settings_io['dataset']['features_dirs']['evaluation'],
            is_training=False,
            settings_data=settings_data,
            settings_io=settings_io)
        
        for pppp in get_clotho_loader(
                settings_io['dataset']['features_dirs']['evaluation'],
                is_training=False,
                settings_data=settings_data,
                settings_io=settings_io):
            logger_main.info("data sample: {}", pppp)
            break

        logger_main.info('Done')

        with no_grad():
            outputs = module_epoch_passing(
                data=validation_data, module=model,
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
        

    def _load_indices_file(settings_files: MutableMapping[str, Any],
                        settings_data: MutableMapping[str, Any]) \
            -> MutableSequence[str]:
        path = Path(
            settings_files['root_dirs']['model_data'],
            settings_files['dataset']['pickle_files_dir'])
        print(settings_files['root_dirs']['model_data'])
        print(settings_files['dataset']['pickle_files_dir'])
        p_field = 'words_list_file_name' \
            if settings_data['output_field_name'].startswith('words') \
            else 'characters_list_file_name'
        indices_filename = path.joinpath(settings_files['dataset']['files'][p_field])
        
        return file_io.load_pickle_file(indices_filename)


    def method(settings: MutableMapping[str, Any]) -> None:
        logger_main = logger.bind(is_caption=False, indent=0)
        logger_main.info('Bootstrapping method')
        pretty_printer = printing.get_pretty_printer()
        logger_inner = logger.bind(is_caption=False, indent=1)

        device, device_name = get_device(0)

        model_dir = Path(
            settings['dirs_and_files']['root_dirs']['outputs'],
            settings['dirs_and_files']['model']['model_dir']
        )

        model_dir.mkdir(parents=True, exist_ok=True)

        logger_inner.info(f'Process on {device_name}\n')

        logger_inner.info('Settings:\n'
                        f'{pretty_printer.pformat(settings)}\n')
        
        model_indices_list = _load_indices_file(
            settings['dirs_and_files'],
            settings['dnn_training_settings']['data'])
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
        _do_inference(
            model=model,
            settings_data=settings['dnn_training_settings']['data'],
            settings_io=settings['dirs_and_files'],
            model_indices_list=model_indices_list)
        logger_inner.info('Inference done')

    prepare_dataset()
    method(config)
    caps.head()
