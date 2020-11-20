from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import datetime
import torch
import git
from inspect import signature
import os
import yaml

class TensorBoardLogger():
    def __init__(self, config):
        self.dummy = config.no_logging
        if config.no_logging:
            print(f"Logging is disabled")
            return
        now = str(datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
        model_name = config.model_type
        self.logs_directory = Path(__file__).parent.absolute().parent / 'logs' / model_name / now
        if not self.logs_directory.exists():
            self.logs_directory.mkdir(parents=True)
        self.initialize_metadata_log(config)
        self.writer = SummaryWriter(str(self.logs_directory))
        self.best_save_metric = None
        self.best_checkpoint_loss = None
        self.best_checkpoint_model_file = None # This will be set the first time we save
        self.last_model_file = None
        self.mid_training_checkpoint = None

    def _dummy(func):
        def dummy_call(*args, **kwargs):
            self = args[0]
            if self.dummy:
                return
            return func(*args, **kwargs)
        return dummy_call


    @_dummy
    def initialize_metadata_log(self, job_configuration):
        repo = git.Repo(search_parent_directories=True)
        config = job_configuration.to_dict()

        self.metadata_file = self.logs_directory / "metadata.yaml"
        self.metadata = {
            **config,
            'git_commit': repo.head.object.hexsha,
            'best_save_metric': None,
            'best_saved_model_file': None,
            'log_directory': str(self.logs_directory),
            'After analysis summary': '*** Fill out this field after training and analyzing ***',
        }
        with open(self.metadata_file, 'w') as f:
            yaml.safe_dump(self.metadata, f)

    @_dummy
    def log_training_loss(self, loss):
        if not hasattr(self, 'training_count'):
            self.training_count = 0
        self.writer.add_scalar("Training loss", loss, self.training_count)
        self.training_count += 1

    @_dummy
    def log_testing_loss(self, loss):
        if not hasattr(self, 'testing_count'):
            self.testing_count = 0
        self.writer.add_scalar("Testing loss", loss, self.testing_count)
        self.testing_count += 1

    @_dummy
    def log_random_evaluation_loss(self, loss):
        # This is not a metric used to train the network because
        # it's too expensive to actually calculate per trajectory
        # But, after training an epoch, it's possible to
        # calculate this metric on a random subsample
        if not hasattr(self, 'eval_count'):
            self.eval_count = 0
        self.writer.add_scalar("Evaluation loss", loss, self.eval_count)
        self.eval_count += 1

    @_dummy
    def log_rollout_MSE(self, loss):
        if not hasattr(self, 'rollout_mse_count'):
            self.rollout_mse_count = 0
        self.writer.add_scalar("Rollout MSE", loss, self.rollout_mse_count)
        self.rollout_mse_count += 1

    @_dummy
    def add_model(self, model, sample_batch_data):
        # The summary writer object requires positional arguments but
        # the code is able to be generic through use of kwargs
        # Here I determine the exact parameters required by this particular
        # model and extract them as required by Tensorboard
        parameters = list(signature(model.forward).parameters.keys())
        parameters = [p for p in parameters if p not in ['args', 'kwargs']]
        data = [sample_batch_data[p] for p in parameters]
        self.writer.add_graph(model, data)

    @_dummy
    def _save(self, epoch, model_file, model, opt, save_info):
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                **save_info,
            },
            model_file,
        )

    @_dummy
    def save_latest_model(self, epoch, pct, model, opt, save_info):
        model_file = self.logs_directory / f"model_ckpt_ep_{epoch}_pct_{pct}.pth"
        self._save(epoch, model_file, model, opt, save_info)
        if self.mid_training_checkpoint is not None:
            # For really big jobs, sometimes the name of the file doesn't change
            # if 1% wasn't achieved in an hour. This would lead to the new file
            # being deleted if we don't check they have the same name
            if (
                model_file != self.mid_training_checkpoint
                and os.path.exists(self.mid_training_checkpoint)
            ):
                os.remove(self.mid_training_checkpoint)
        self.mid_training_checkpoint = model_file


    @_dummy
    def save_model(self, epoch, model, opt, save_info, metric):
        """
        This function will always save the most recent model,
        but will also do some cleanup and global accounting for saved models

        Models that will be saved:
        - Within every 10 epochs (i.e. 0-9, 10-19, etc), it will maintain the
        model with the best value for "metric"
        - The most recent epoch's model

        This function expects an integer value for the epoch, but could be used
        within epochs as well. If it's a checkpoint epoch, it will just
        overwrite the previous save for that epoch. If it is not a checkpoint
        epoch, it'll follow the same logic ranking by "metric"

        Parameters
        ----------
        epoch : The epoch being saved
        model : The current model
        opt : The current optimizer
        save_info : Extra info to save (such as metrics)
        metric : The metric to use for deciding whether to checkpoint

        Returns
        -------
        void

        """
        model_file = self.logs_directory / f"model_epoch_{epoch}.pth"

        if 'save_metric' in save_info and save_info['save_metric'] != metric:
            print(
                f"Logger received conflicting save metric values: "
                f"{save_info['save_metric']} vs {metric}. "
                f"Choosing the latter, but former will be saved "
                f"with key \"ignored_save_metric\"."
            )
            save_info["ignored_save_metric"] = save_info['save_metric']
        save_info['save_metric'] = metric

        self._save(epoch, model_file, model, opt, save_info)

        # This resets the model tracking every ten epochs
        # to ensure that we are saving checkpoints
        if epoch % 10 == 0:
            checkpoint_save = True
        else:
            checkpoint_save = False

        # Only delete previous best file if we aren't checkpointing
        if checkpoint_save:
            print(f"Starting new checkpoint group at: {model_file}")
        elif metric < self.best_checkpoint_loss:
            # At a checkpoint, we save a new model, but at a non-checkpoint,
            # overwrite the old, best model (within that checkpoint range)
            print(f"Updated model checkpoint will be saved to: {model_file}")
            if os.path.exists(self.best_checkpoint_model_file):
                os.remove(self.best_checkpoint_model_file)

        # Only delete the previous checkpoint file if it's not the previous best checkpoint file
        if (
            self.best_checkpoint_model_file is not None
            and self.last_model_file is not None
            and self.best_checkpoint_model_file.resolve() != self.last_model_file.resolve()
        ):
            if os.path.exists(self.last_model_file):
                os.remove(self.last_model_file)
        self.last_model_file = model_file

        # Store the current checkpoints in the class
        if checkpoint_save or metric < self.best_checkpoint_loss:
            self.best_checkpoint_loss = metric
            self.best_checkpoint_model_file = self.logs_directory / f"model_epoch_{epoch}.pth"
            self.last_model_file = self.best_checkpoint_model_file

        # This does global accounting over all the saved models so far
        if self.best_save_metric is None or metric < self.best_save_metric:
            print(f"Updating metadata with new best model: {self.metadata_file}")
            self.best_save_metric = metric
            self.metadata['best_save_metric'] = str(metric)
            self.metadata['best_saved_model_file'] = str(self.best_checkpoint_model_file)
            with open(self.metadata_file, 'w') as f:
                yaml.safe_dump(self.metadata, f)
