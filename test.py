import argparse
import json
from pathlib import Path
from typing import Dict

import torch

import lib.model as module_model
import lib.storage as module_storage
from lib.config_processing import ConfigParser
from lib.storage.experiments_storage import ExperimentsStorage
from lib.storage.external_storage import ExternalStorage


BEST_CHECKPOINT = {
    'exp_name': 'baseline (sinc-no-abs)',
    'run_name': '1216_102848',
    'checkpoint_name': 'model_best',
}


def main(model_config: Dict, checkpoint_filepath: Path,
         input_dirpath: Path, output_dirpath: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Loading weights of the model...')
    model = ConfigParser.init_obj(model_config, module_model)
    checkpoint = torch.load(checkpoint_filepath, map_location=device)["state_dict"]
    model.load_state_dict(checkpoint)
    print('Weights loaded')


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
        "-o",
        "--output",
        default='result',
        type=str,
        help="a directory with resulting audios",
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
    model_config_path = Path(args.checkpoint).parent / "config.json"
    with model_config_path.open() as f:
        model_config = json.load(f)['arch']

    main(model_config, args.checkpoint, args.input, args.output)
