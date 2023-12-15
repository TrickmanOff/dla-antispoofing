import random
from collections import defaultdict
from pathlib import Path
from random import shuffle
from typing import Dict, Sequence

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from lib.metric.base_metric import BaseMetric
from lib.metric.utils import basic_plot_eer
from lib.trainer.base_trainer import BaseTrainer
from lib.utils import inf_loop, MetricTracker, get_lr, open_image_of_pyplot


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics: Sequence[BaseMetric],
            optimizer,
            config,
            device,
            dataloaders,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device, lr_scheduler)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler = lr_scheduler
        self.log_step = config["trainer"].get("log_step", 50)

        self.train_metrics = MetricTracker(
            "loss", "grad norm", *[m.name for m in self.metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", *[m.name for m in self.metrics], writer=self.writer
        )
        self.accumulated_grad_steps = 0
        self.accumulate_grad_steps = config["trainer"].get("accumulate_grad_steps", 1)
        self.accumulated_batches_data_keys = {"pred_logits", "is_bonafide"}

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["wave", "is_bonafide"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch) -> dict:
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)

        accumulated_batches = defaultdict(list)  # only simple concatenation of tensors along the 1st dim is currently supported
        accumulated_new_values_cnt = defaultdict(int)  # {metric_name: cnt}

        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
                for key in self.accumulated_batches_data_keys:
                    if key in batch:
                        value = batch[key]
                        if isinstance(value, torch.Tensor):
                            value = value.detach().cpu()
                        accumulated_batches[key].append(value)
                    for met in self.metrics:
                        accumulated_new_values_cnt[met.name] += batch['pred_logits'].shape[0]
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx != 0 and batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx + 1)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", get_lr(self.optimizer) if self.lr_scheduler is None else self.lr_scheduler.get_last_lr()[0]
                )
                self._log_predictions(**batch)

                # a bit clumsy though better than ad-hoc for one metric
                # I decided not to spend a lot of time for ONE metric in this homework
                met_outputs = {}
                for met in self.metrics:
                    if met.calc_on_train and \
                            met.calc_on_entire_dataset and \
                            met.calc_on_entire_dataset_train_accum != -1 and \
                            (met.calc_on_entire_dataset_train_accum is None or
                                    accumulated_new_values_cnt[met.name] >= met.calc_on_entire_dataset_train_accum):
                        for key in accumulated_batches:
                            if len(accumulated_batches[key]) > 1:
                                accumulated_batches[key] = [torch.concat(accumulated_batches[key], dim=0)]
                        print(f'Accumulated {accumulated_new_values_cnt[met.name]} new values for metric "{met.name}"')
                        st_index = -accumulated_new_values_cnt[met.name]
                        accumulated_new_values_cnt[met.name] = 0
                        met_kwargs = {key: values[0][st_index:] for key, values in accumulated_batches.items()}
                        met_output = met(**met_kwargs)
                        self.train_metrics.update(met.name, met_output.pop('metric') if isinstance(met_output, dict) else met_output)
                        if isinstance(met_output, dict):
                            met_outputs.update(met_output)

                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
                self._log_metrics_outputs(met_outputs)
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        # ----
        for key in accumulated_batches:
            if len(accumulated_batches[key]) > 1:
                accumulated_batches[key] = [torch.concat(accumulated_batches[key], dim=0)]
        met_outputs = {}
        for met in self.metrics:
            if met.calc_on_train and met.calc_on_entire_dataset and met.calc_on_entire_dataset_train_accum == -1:
                met_kwargs = {key: values[0] for key, values in accumulated_batches.items()}
                met_output = met(**met_kwargs)
                self.train_metrics.update(met.name, met_output.pop('metric') if isinstance(met_output, dict) else met_output)
                if isinstance(met_output, dict):
                    met_outputs.update(met_output)
        self._log_scalars(self.train_metrics)
        self.train_metrics.reset()
        self._log_metrics_outputs(met_outputs)
        # ----

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        outputs = self.model(**batch)
        if is_train and self.accumulated_grad_steps == 0:
            self.optimizer.zero_grad()
        if type(outputs) is dict:
            batch.update(outputs)
        else:
            batch["pred_logits"] = outputs

        batch["loss"] = self.criterion(**batch)
        if is_train:
            (batch["loss"] / self.accumulate_grad_steps).backward()
            self.accumulated_grad_steps += 1
            if self.accumulated_grad_steps % self.accumulate_grad_steps == 0:
                self._clip_grad_norm()
                self.optimizer.step()
                self.accumulated_grad_steps = 0
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

        metrics.update("loss", batch["loss"].item())
        for met in self.metrics:
            if not met.calc_on_entire_dataset:
                metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()

        accumulated_batches = defaultdict(list)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
                for key in self.accumulated_batches_data_keys:
                    if key in batch:
                        value = batch[key]
                        if isinstance(value, torch.Tensor):
                            value = value.detach().cpu()
                        accumulated_batches[key].append(value)

            for key in accumulated_batches:
                accumulated_batches[key] = torch.concat(accumulated_batches[key], dim=0)
            met_outputs = {}
            for met in self.metrics:
                if met.calc_on_non_train and met.calc_on_entire_dataset:
                    met_output = met(**accumulated_batches)
                    self.evaluation_metrics.update(met.name,
                                                   met_output.pop('metric') if isinstance(met_output, dict) else met_output)
                    if isinstance(met_output, dict):
                        met_outputs.update(met_output)

            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_metrics_outputs(met_outputs)
            self._log_predictions(**batch)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_metrics_outputs(self, metrics_outputs: Dict):
        if 'frr' in metrics_outputs:
            assert 'far' in metrics_outputs and 'eer' in metrics_outputs
            print('plotting EER')
            eer_fig = basic_plot_eer(metrics_outputs['eer'], metrics_outputs['frr'], metrics_outputs['far'])
            with open_image_of_pyplot(eer_fig) as img_filepath:
                self.writer.add_image('EER_plot', img_filepath)

    def _log_predictions(
            self,
            id,
            spoofing_algo,
            wave,
            wave_length,
            is_bonafide,
            pred_logits,
            examples_to_log=10,
            *args,
            **kwargs,
    ):
        if self.writer is None:
            return

        pred_probs = F.softmax(pred_logits.detach().cpu(), dim=-1)  # (B, 2)
        pred_bonafide_probs = pred_probs[:, 1]  # (B,)

        tuples = list(zip(id, spoofing_algo, wave, wave_length, is_bonafide, pred_bonafide_probs))
        rows = {}
        for id, spoofing_algo, wave, wave_length, is_bonafide, pred_bonafide_prob \
                in tuples[:examples_to_log]:
            audio = self.writer.create_audio(wave.detach().cpu().squeeze()[:wave_length],
                                             sample_rate=self.config["preprocessing"]["sr"])
            rows[id] = {
                "is bonafide": is_bonafide.item(),
                "bonafide pred prob": pred_bonafide_prob.item(),
                "spoofing algo": spoofing_algo,
                "audio": audio
            }
        table = pd.DataFrame.from_dict(rows, orient="index")\
                            .reset_index().rename(columns={'index': 'id'})
        self.writer.add_table("predictions", table)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            if metric_tracker.count(metric_name) == 0:
                continue
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
