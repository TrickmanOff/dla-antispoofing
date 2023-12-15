import logging
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from lib.config_processing import ConfigParser

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
            self,
            index: List[Dict[str, Any]],
            config_parser: ConfigParser,
            is_train: bool,
            wave_augs=None,
            limit: Optional[int] = None,
            max_audio_length: Optional[float] = None,
            same_audio_length: Optional[float] = None,
    ):
        """
        :param index: of format [{'path': <path to the audio>, 'is_bonafide': <1 if audio is bona fide, 0 otherwise}, ...]
            the 'is_bonafide' flag is optional
        :param limit: not more than `limit` random audios are taken from the index
        :param max_audio_length: max audio length in seconds
                                 all audios with length greater will be excluded
        :param same_audio_length: if specified, then all audios will have the same length (in seconds)
            longer audios will be cropped (each time a random part is selected),
            shorter ones will be concatenated with themselves
        """
        self.is_train = is_train
        self.config_parser = config_parser
        self.wave_augs = wave_augs
        self.same_audio_length = None if same_audio_length is None else int(self.config_parser["preprocessing"]["sr"] * same_audio_length)

        self._assert_index_is_valid(index)
        index = self._filter_records_from_dataset(index, max_audio_length, limit)
        # it's a good idea to sort index by audio length
        # It would be easier to write length-based batch samplers later
        # index = self._sort_index(index)
        self._index = index

    def __getitem__(self, ind: int) -> Dict[str, Any]:
        data_dict = self._index[ind]
        audio_path = data_dict["path"]
        audio_wave = self.load_audio(audio_path)
        audio_wave = self.process_wave(data_dict["id"], audio_wave)
        entry = {
            "wave": audio_wave,
            "duration": audio_wave.size(1) / self.config_parser["preprocessing"]["sr"],
            "audio_path": audio_path,
        }
        for key in ["is_bonafide", "id", "speaker_id", "spoofing_algo"]:
            if key in data_dict:
                entry[key] = data_dict[key]
        return entry

    @staticmethod
    def _sort_index(index: List[Dict[str, Any]]):
        return sorted(index, key=lambda x: x["audio_len"])

    @staticmethod
    def fix_length(target_length: int, wave: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        """
        :param target_length: the number of samples
        :param wave: of shape (1, T)
        :return: wave of shape (1, target_length)
        """
        wave_len = wave.shape[1]
        if wave_len < target_length:
            times = (target_length + wave_len - 1) // wave_len
            wave = wave.repeat((1, times))[:, :target_length]
        else:
            if seed is None:
                st = random.randint(0, wave_len - target_length)
            else:
                st = abs(2*seed + 42) % (wave_len - target_length + 1)
            wave = wave[:, st:st+target_length]
        return wave

    def __len__(self):
        return len(self._index)

    def load_audio(self, path: str):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def process_wave(self, audio_id: str, audio_tensor_wave: Tensor):
        with torch.no_grad():
            if self.wave_augs is not None:
                audio_tensor_wave = self.wave_augs(audio_tensor_wave)
            if self.same_audio_length is not None:
                seed = None if self.is_train else hash(audio_id)
                audio_tensor_wave = self.fix_length(self.same_audio_length, audio_tensor_wave, seed=seed)
            return audio_tensor_wave

    @staticmethod
    def _filter_records_from_dataset(
            index: List[Dict[str, Any]], max_audio_length: Optional[float], limit: Optional[int]
    ) -> List:
        initial_size = len(index)
        if max_audio_length is not None:
            print("Filtering audios by their length...")
            for el in tqdm(index):
                wave, sr = torchaudio.load(el["path"])
                el["audio_len"] = wave.shape[1] / sr
            exceeds_audio_length = np.array([el["audio_len"] for el in index]) >= max_audio_length
            _total = exceeds_audio_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_audio_length} seconds. Excluding them."
            )
        else:
            exceeds_audio_length = False

        initial_size = len(index)

        records_to_filter = exceeds_audio_length

        if records_to_filter is not False and records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(
                f"Filtered {_total}({_total / initial_size:.1%}) records  from dataset"
            )

        if limit is not None:
            random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        return index

    @staticmethod
    def _assert_index_is_valid(index: List[Dict[str, Any]]):
        for entry in index:
            assert "path" in entry, (
                "Each dataset item should include field 'path'" " - path to audio file."
            )
