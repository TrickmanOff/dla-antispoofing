import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from tqdm import tqdm

from lib.datasets.base_dataset import BaseDataset
from lib.config_processing import ConfigParser
from lib.utils.util import download_file


logger = logging.getLogger(__name__)

URL_LINKS = {
    "LA": "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y",
}

PARTS_SUBDIRS = {
    "train": "ASVspoof2019_LA_train",
    "eval": "ASVspoof2019_LA_eval",
    "dev": "ASVspoof2019_LA_dev",
}


class ASVSpoof2019LADataset(BaseDataset):
    def __init__(self, part: str, config_parser: ConfigParser,
                 data_dir: Optional[Union[str, Path]] = None,
                 index_dir: Optional[Union[str, Path]] = None,
                 *args, **kwargs):
        assert part in PARTS_SUBDIRS

        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        if isinstance(index_dir, str):
            index_dir = Path(index_dir)

        if data_dir is None:
            data_dir = config_parser.get_data_root_dir() / "data" / "datasets" / "ASVSpoof2019"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        self._index_dir = data_dir if index_dir is None else index_dir
        index = self._get_or_load_index(part)

        super().__init__(index, *args, config_parser=config_parser, is_train=(part == 'train'), **kwargs)

    def _load_dataset(self):
        arch_path = self._data_dir / f"LA.zip"
        print(f"Loading LA dataset...")
        download_file(URL_LINKS["LA"], arch_path.parent, arch_path.name)
        print(f"Unpacking LA dataset...")
        shutil.unpack_archive(arch_path, self._data_dir)
        os.remove(str(arch_path))

    def _get_or_load_index(self, part: str):
        self._index_dir.mkdir(parents=True, exist_ok=True)
        index_path = self._index_dir / f"{part}_index.json"

        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part: str):
        index = []
        split_dir = self._data_dir / "LA" / PARTS_SUBDIRS[part] / "flac"
        if not split_dir.exists():
            self._load_dataset()

        table_path = self._data_dir / "LA" / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{part}.{'trn' if part == 'train' else 'trl'}.txt"
        provided_index = pd.read_csv(table_path, sep=' ', header=None, names=["speaker_id", "audio_id", "sth", "algo", "type"])
        provided_data = {
            row["audio_id"]: {
                "speaker_id": row["speaker_id"],
                "is_bonafide": 1 if row["type"] == "bonafide" else 0,
                "spoofing_algo": row["algo"],
            }
            for _, row in provided_index.iterrows()
        }

        audio_pattern = re.compile(r'^LA_.*_\d+\.flac$')
        for flac_filename in tqdm(
                os.listdir(split_dir), desc=f"Scanning the '{part}' part of the dataset"
        ):
            flac_filepath = split_dir / flac_filename
            if not audio_pattern.match(flac_filename):
                continue
            audio_id = flac_filepath.stem
            entry = {
                "id": audio_id,
                "path": str(flac_filepath),
            }
            entry.update(provided_data[audio_id])
            index.append(entry)

        return index
