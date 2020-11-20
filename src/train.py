import src.loss as loss
import torch
import numpy as np
from tqdm.auto import tqdm
from inspect import getfullargspec
from src.utils import send_batch_to_device
from datetime import datetime
import gc
import torch.multiprocessing as mp
import torch.distributed as dist
from src.data_loader import distribute_dataloader
from src import logging
from pytorch_model_summary import summary


class Trainer:
    def __init__(self, mdl, loss_func, multigpu):
        self.mdl = mdl
        self.loss_func = loss_func
        self.checkpoint_time = None  # Initialized in the run function
        self.multigpu = multigpu

    def fit(self, train_dl, test_dl, eval_dl, config):
        if self.multigpu:
            mp.spawn(
                Trainer._run_worker,
                nprocs=torch.cuda.device_count(),
                args=((
                    self.mdl,
                    train_dl,
                    test_dl,
                    eval_dl,
                    self.loss_func,
                    config,
                )),
            )
        else:
            self._run_single_process(
                self.mdl,
                train_dl,
                test_dl,
                eval_dl,
                self.loss_func,
                config,
            )

    @staticmethod
    def get_device_keys(fn):
        device_keys = [k for k in getfullargspec(fn)[0] if k != 'self']
        return tuple(device_keys + ['supervision'])

    @staticmethod
    def summarize(mdl, sample_input, gpu_id=None):
        return
        # TODO fix this section
        device_keys = Trainer.get_device_keys(mdl.forward)[:-1]
        if gpu_id == None:
            sample_input = [sample_input[key].shape[1:] for key in device_keys]
        else:
            sample_input = [sample_input[key].shape[1:] for key in device_keys]
        logger.add_model(mdl, sample_input)
        print(summary(mdl, sample_input))

    @staticmethod
    def _run_worker(gpu_id, mdl, train_dl, test_dl, eval_dl, loss_func, config):
        # initialize the process group
        if gpu_id == 0:
            logger = logging.TensorBoardLogger(config)
        else:
            logger = None
        torch.cuda.set_device(gpu_id)
        params = config.multigpu_parameters
        world_size = params.ngpus_per_node * params.num_nodes
        global_gpu_id = params.node_idx * params.ngpus_per_node + gpu_id
        dist.init_process_group(
            params.backend,
            rank=global_gpu_id,
            world_size=world_size,
            init_method=params.url,
        )
        mdl = mdl.to(gpu_id)
        ddp_mdl = torch.nn.parallel.DistributedDataParallel(
            mdl,
            device_ids=[gpu_id],
            output_device=gpu_id,
        )
        opt = mdl.optimizer(ddp_mdl.parameters())
        train_dl = distribute_dataloader(train_dl, world_size, global_gpu_id)
        test_dl = distribute_dataloader(test_dl, world_size, global_gpu_id)
        eval_dl = distribute_dataloader(eval_dl, world_size, global_gpu_id)
        if gpu_id == 0:
            sample_input = next(train_dl.__iter__())
            Trainer.summarize(mdl, sample_input, 0)
        for epoch in range(config.epochs):
            train_dl.sampler.set_epoch(epoch)
            test_dl.sampler.set_epoch(epoch)
            eval_dl.sampler.set_epoch(epoch)
            train_loss = Trainer.train_loop(
                mdl,
                opt,
                epoch,
                train_dl,
                loss_func,
                logger,
                config,
            )
            if gpu_id == 0:
                test_loss = self.test_loop(mdl, epoch, test_dl, config)
                eval_loss = Trainer.evaluation_loop(
                    mdl,
                    epoch,
                    eval_dl,
                    loss_func,
                    logger,
                    config,
                )
                logger.save_model(
                    epoch,
                    mdl,
                    opt,
                    save_info={
                        'train_loss': train_loss,
                        'test_loss': test_loss,
                        'evaluation_loss': eval_loss,
                    },
                    metric=eval_loss,
                )

    @staticmethod
    def _run_single_process(mdl, train_dl, test_dl, eval_dl, loss_func, config):
        if config.cpu:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
        mdl.to(device)
        opt = mdl.optimizer()

        logger = logging.TensorBoardLogger(config)

        sample_input = next(train_dl.__iter__())
        Trainer.summarize(mdl, sample_input, device)

        for epoch in range(config.epochs):
            train_loss = Trainer.train_loop(
                mdl,
                opt,
                epoch,
                train_dl,
                loss_func,
                logger,
                config,
            )
            test_loss = Trainer.test_loop(
                mdl,
                epoch,
                test_dl,
                loss_func,
                logger,
                config,
            )
            eval_loss = Trainer.evaluation_loop(
                mdl,
                epoch,
                eval_dl,
                loss_func,
                logger,
                config,
            )
            logger.save_model(
                epoch,
                mdl,
                opt,
                save_info={
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'evaluation_loss': eval_loss,
                },
                metric=eval_loss,
            )

    @staticmethod
    def train_loop(
        mdl,
        opt,
        epoch,
        train_dl,
        loss_func,
        logger,
        config,
    ):
        device = next(mdl.parameters()).device
        mdl.train()
        assert not mdl.in_rollout_mode()
        progress = tqdm(train_dl, position=device.index)
        progress.set_description(
            f"Process {device.index}, Epoch {epoch}/{config.epochs}, Training"
        )
        train_losses, train_batch_sizes = [], []
        checkpoint_time = datetime.now()
        device_keys = Trainer.get_device_keys(mdl.forward)
        for i, batch in enumerate(progress):
            device_batch = send_batch_to_device(batch, device_keys, device)
            x_batch = {k: device_batch[k] for k in device_batch if k != 'supervision'}
            y_batch = mdl.transform(device_batch['supervision'])
            l, s = loss.loss_batch(mdl, loss_func, x_batch, y_batch, opt)
            train_losses.append(l)
            train_batch_sizes.append(s)
            current_time = datetime.now()
            if (
                device.index == 0
                and (current_time - checkpoint_time).total_seconds() > config.checkpoint_interval
            ):
                logger.save_latest_model(
                    epoch,
                    int(100 * i / progress.total),
                    mdl,
                    opt,
                    {'batch_loss': l, 'batch_size': s},
                )
                checkpoint_time = current_time
                logger.log_training_loss(l / s)
            gc.collect()
        loss_val = Trainer.weighted_loss(np.array(train_losses), np.array(train_batch_sizes))
        print(f"Epoch {epoch} training loss: {loss_val}")
        return loss_val

    @staticmethod
    def test_loop(mdl, epoch, test_dl, loss_func, logger, config):
        device = next(mdl.parameters()).device
        assert device.index == 0
        mdl.eval()
        assert not mdl.in_rollout_mode()
        device_keys = Trainer.get_device_keys(mdl.forward)
        with torch.no_grad():
            progress = tqdm(test_dl, position=device.index)
            progress.set_description(
                f"Process {device.index}, Epoch {epoch}/{config.epochs}, Testing"
            )
            test_losses = []
            test_batch_sizes = []
            for i, batch in enumerate(progress):
                device_batch = send_batch_to_device(
                    batch,
                    device_keys,
                    device,
                )
                x_batch = {
                    k: device_batch[k]
                    for k in device_batch if k != 'supervision'
                }
                y_batch = mdl.transform(device_batch['supervision'])
                l, s = loss.loss_batch(mdl, loss_func, x_batch, y_batch)
                test_losses.append(l)
                test_batch_sizes.append(s)
        loss_val = Trainer.weighted_loss(
            np.array(test_losses), np.array(test_batch_sizes)
        )
        logger.log_testing_loss(loss_val.item())
        print(f"Epoch {epoch} validation loss: {loss_val}")
        return loss_val

    @staticmethod
    def evaluation_loop(mdl, epoch, eval_dl, loss_func, logger, config):
        device = next(mdl.parameters()).device
        assert device.index == 0
        # Keep the faith bro!
        mdl.eval()
        mdl.set_rollout_mode()
        device_keys = Trainer.get_device_keys(mdl.forward)
        with torch.no_grad():
            total = min(1000 // config.batch_size + 1, len(eval_dl))
            progress = tqdm(eval_dl, total=total, position=device.index)
            progress.set_description(
                f"Process {device.index}, Epoch {epoch}/{config.epochs}, Evaluating with rollout"
            )
            eval_losses = []
            eval_batch_sizes = []
            num_processed = 0
            for i, batch in enumerate(progress):
                device_batch = send_batch_to_device(batch, device_keys, device)
                x_batch = {k: device_batch[k] for k in device_batch if k != 'supervision'}
                y_batch = mdl.transform(device_batch['supervision'])
                l, s = loss_func(mdl(**x_batch), y_batch), len(next(iter(x_batch.values())))
                eval_losses.append(l)
                eval_batch_sizes.append(s)
                num_processed += s
                if num_processed > 1000:
                    break
            loss_val = Trainer.weighted_loss(np.array(eval_losses), np.array(eval_batch_sizes))
        logger.log_random_evaluation_loss(loss_val.item())
        print(
            f"Epoch {epoch} random mean evaluation error: "
            f"{loss_val}"
        )
        mdl.unset_rollout_mode()
        return loss_val

    @staticmethod
    def weighted_loss(losses, batch_sizes):
        weighted_losses = np.multiply(losses, batch_sizes)
        return np.sum(weighted_losses) / np.sum(batch_sizes)
