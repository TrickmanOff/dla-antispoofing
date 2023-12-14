from typing import Optional


class BaseMetric:
    def __init__(self, name=None,
                 calc_on_train: bool = True,
                 calc_on_non_train: bool = True,
                 calc_on_entire_dataset: bool = False,
                 calc_on_entire_dataset_train_accum: Optional[int] = None,
                 *args, **kwargs):
        """
        :param calc_on_train_accum: is used only if `calc_on_entire_dataset` is set to True
            if None, then all values collected during an epoch will be used
            else when `calc_on_train_accum` values are accumulated, the metric is calculated
        """
        self.name = name if name is not None else type(self).__name__
        self.calc_on_train = calc_on_train
        self.calc_on_non_train = calc_on_non_train
        self.calc_on_entire_dataset = calc_on_entire_dataset
        self.calc_on_entire_dataset_train_accum = calc_on_entire_dataset_train_accum

    def __call__(self, **batch):
        raise NotImplementedError()
