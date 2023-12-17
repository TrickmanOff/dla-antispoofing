import dataclasses
import json
import random
import tempfile
from collections import OrderedDict
from contextlib import contextmanager
from itertools import repeat
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import Tensor
from tqdm import tqdm

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent


def align_last_dim(x: Tensor, target: Tensor, padding_value: float = 0.):
    target_T = target.shape[-1]
    T = x.shape[-1]
    if target_T < T:
        return x[..., :target_T]
    else:
        return F.pad(x, (0, target_T - T), value=padding_value)


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        elif isinstance(o, Path):
            return str(o)
        return super().default(o)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        # if self.writer is not None:
        #     self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def count(self, key):
        return self._data.counts[key]

    def result(self):
        return dict(self._data.average)

    def keys(self):
        return self._data.total.keys()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def download_file(url, to_dirpath=None, to_filename=None):
    local_filename = to_filename or url.split('/')[-1]
    if to_dirpath is not None:
        to_dirpath.mkdir(exist_ok=True, parents=True)
        local_filename = to_dirpath / local_filename
    chunk_size = 2**20  # in bytes
    with requests.get(url, stream=True) as r:
        if 'Content-length' in r.headers:
            total_size = int(r.headers['Content-length'])
            total = (total_size - chunk_size + 1) // chunk_size
        else:
            total_size = None
        desc = f'Downloading file'
        if total_size is not None:
            desc += f', {total_size / (2**30):.2f}GBytes'
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=chunk_size), total=total, desc=desc, unit='MBytes'):
                f.write(chunk)
    return local_filename


@contextmanager
def open_image_of_pyplot(figure) -> str:
    file = tempfile.NamedTemporaryFile()
    figure.savefig(file, format='png', bbox_inches='tight')
    plt.close()

    try:
        yield file.name
    finally:
        file.close()


def fix_audio_length(target_length: int, wave: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
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
            print(wave_len, target_length, seed)
            st = abs(2*seed + 42) % (wave_len - target_length + 1)
            print(st)
        wave = wave[:, st:st+target_length]
    return wave
