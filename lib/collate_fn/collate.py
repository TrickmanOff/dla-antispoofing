import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor, LongTensor


logger = logging.getLogger(__name__)


PADDING_VALUE = 0


def pad_last_d(input: List[Tensor], padding_value: float = PADDING_VALUE) -> Tuple[Tensor, LongTensor]:
    """
    each of B inputs is of shape (..., S_i)

    result:
        stack:  (B, ..., max_i S_i)
        length: (B,) - initial lengths of each sequence
    """
    length = LongTensor([x.shape[-1] for x in input])
    max_len = length.max()

    shape = [len(input)] + list(input[0].shape)
    shape[-1] = max_len

    stack = torch.full(shape, padding_value, dtype=input[0].dtype)

    for i, x in enumerate(input):  # (..., S_i)
        stack[i, ..., :x.shape[-1]] = x

    return stack, length


def collate_fn(dataset_items: List[dict]) -> Dict[str, Any]:
    """
    Collate and pad fields in dataset items
    """
    all_items = defaultdict(list)  # {str: [val1, val2, ...], ...}
    all_keys = next(iter(dataset_items)).keys()
    for items in dataset_items:
        assert all_keys == items.keys(), f'Keys for items are not the same: {items.keys()} != {all_keys}'
        for key, val in items.items():
            all_items[key].append(val)

    result_batch = {}

    # some extra info
    for key in ['id', 'speaker_id', 'spoofing_algo']:
        if key in all_items:
            result_batch[key] = all_items[key]

    # wave
    result_batch['wave'], result_batch['wave_length'] = pad_last_d(all_items['wave'])

    # is_bonafide
    result_batch['is_bonafide'] = torch.LongTensor(all_items['is_bonafide'])

    return result_batch
