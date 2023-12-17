import argparse
import json
import os
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import wandb
from tqdm import tqdm

import lib.model as module_model
import lib.storage as module_storage
from lib.config_processing import ConfigParser
from lib.storage.experiments_storage import ExperimentsStorage
from lib.storage.external_storage import ExternalStorage
from lib.utils import fix_audio_length


BEST_CHECKPOINT = {
    'exp_name': 'baseline (sinc-no-abs)',
    'run_name': '1216_102848',
    'checkpoint_name': 'model_best',
}


def main(training_config: Dict, checkpoint_filepath: Path,
         input_dirpath: Path,
         log_results_to_wandb: bool = True):
    if log_results_to_wandb:
        wandb.login()
        wandb.init(
            project='dla-antispoof',
            name='inference',
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Loading weights of the model...')
    model = ConfigParser.init_obj(training_config['arch'], module_model)
    checkpoint = torch.load(checkpoint_filepath, map_location=device)["state_dict"]
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print('Weights loaded')

    training_sr = training_config['preprocessing']['sr']
    same_audio_len = training_config['data']['train']['datasets'][0]['args'].get('same_audio_length', None)
    if same_audio_len is not None:
        print(f'Note that all audios will be cropped or duplicated to be {same_audio_len} seconds long')

    rows = []
    with torch.no_grad():
        for audio_filename in tqdm(sorted(os.listdir(input_dirpath)), desc='Processing audios'):
            wave, sr = torchaudio.load(input_dirpath / audio_filename)
            # assert sr == training_sr, f'The model was trained with sample rate = {training_sr}, not {sr}'
            if sr != training_sr:
                wave = torchaudio.functional.resample(wave, sr, training_sr)
            # waves = {audio_filename + ' (full)': wave}
            waves = {}
            wave_name = audio_filename

            if same_audio_len is not None:
                if wave.shape[-1] < int(same_audio_len * training_sr):
                    wave_name += ' (duplicated)'
                if wave.shape[-1] > int(same_audio_len * training_sr):
                    wave_name += ' (cropped)'
                waves[wave_name] = fix_audio_length(int(same_audio_len * training_sr), wave, seed=hash(audio_filename))
            for wave_name, wave in waves.items():
                pred_logits = model(wave=wave.unsqueeze(0).to(device), wave_length=torch.LongTensor([wave.shape[-1]]).to(device))[0]  # (2)
                pred_probs = F.softmax(pred_logits.cpu(), dim=0)
                row = {
                    'audio_filename': wave_name,
                    'pred probability of being bonafide': pred_probs[1].item(),
                }
                if log_results_to_wandb:
                    row['audio'] = wandb.Audio(wave.numpy(), sample_rate=training_sr)
                rows.append(row)

    res_df = pd.DataFrame(rows).set_index('audio_filename')
    if log_results_to_wandb:
        wandb.log({'results': wandb.Table(dataframe=res_df.reset_index())})
    print(res_df)


def get_best_checkpoint_filepath(external_storage_config: Dict) -> Path:
    exps_storage = ExperimentsStorage('saved/models')
    run_storage = exps_storage.get_run(exp_name=BEST_CHECKPOINT['exp_name'],
                                       run_name=BEST_CHECKPOINT['run_name'],
                                       create_run_if_no=True)
    if BEST_CHECKPOINT['checkpoint_name'] not in run_storage.get_checkpoints_filepaths():
        external_storage: ExternalStorage = ConfigParser.init_obj(external_storage_config, module_storage)
        print('Importing the best model...')
        external_storage.import_checkpoint(run_storage, BEST_CHECKPOINT['checkpoint_name'])
        external_storage.import_config(run_storage)
    return run_storage.get_checkpoints_filepaths()[BEST_CHECKPOINT['checkpoint_name']]


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Text AntiSpoofing inference script")
    args.add_argument(
        "-i",
        "--input",
        default='demo_audios',
        type=str,
        help="a directory with audios",
    )
    args.add_argument(
        "-c",
        "--checkpoint",
        default='',
        type=str,
        help="model checkpoint filepath",
    )
    args = args.parse_args()

    # download the best checkpoint if not specified
    if args.checkpoint == '':
        external_storage_config = json.load(open('gdrive_storage/external_storage.json', 'r'))['external_storage']
        args.checkpoint = get_best_checkpoint_filepath(external_storage_config)

    # we assume it is located with checkpoint in the same folder
    training_config_path = Path(args.checkpoint).parent / "config.json"
    with training_config_path.open() as f:
        training_config = json.load(f)

    main(training_config, args.checkpoint, Path(args.input))
