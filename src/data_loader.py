from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import enum
import h5py

def worker_init_fn(worker_id):
    """ Use this to bypass issue with PyTorch dataloaders using deterministic RNG for Numpy
        https://github.com/pytorch/pytorch/issues/5059
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def distribute_dataloader(dl, num_replicas, rank):
    sampler = torch.utils.data.distributed.DistributedSampler(
        dl.dataset,
        num_replicas,
        rank,
        shuffle=isinstance(dl.sampler, torch.utils.data.sampler.RandomSampler)
    )
    return DataLoader(
        dataset=dl.dataset,
        batch_size=dl.batch_size,
        shuffle=False,
        num_workers=dl.num_workers,
        collate_fn=dl.collate_fn,
        pin_memory=dl.pin_memory,
        drop_last=dl.drop_last,
        timeout=dl.timeout,
        worker_init_fn=dl.worker_init_fn,
        sampler=sampler,
        # In the docs, there is also something called multiprocessing_context but that doesn't seem to be used anywhere
    )


class DatasetType(enum.Enum):
    TRAIN = 0
    TEST = 1


class FishboticsBaseDataset(Dataset):
    def __init__(
            self,
            directory,
            acquire_test_dataset=False,
            mini=False,
            **kwargs,
    ):
        self.mini = mini
        self._init_directory(
            directory,
            acquire_test_dataset,
        )
        # TODO more stuff goes here if you need it

    def _init_directory(self, directory, acquire_test_dataset):
        if acquire_test_dataset:
            self.type = DatasetType.TEST
            directory = directory / 'test'
        else:
            self.type = DatasetType.TRAIN
            directory = directory / 'train'

        # TODO uncomment if using hdf5
        # databases = list(directory.glob('**/*.hdf5'))
        # num_elements = []
        # for filename in databases:
        #     with h5py.File(str(filename), 'r') as f:
        #         # The goal direction dataset has dimension (num_trajectories, 3)
        #         num_elements.append(f['INSERT KEY HERE'].shape[0])
        # self._databases = list(zip(num_elements, databases))

    def get_file_and_index(self, global_index):
        # Entire dataset can be spread over multiple files (it is sometimes
        # faster to generate the dataset this way).
        # We need to figure out which file to access and get the index
        # within that file
        local_index = global_index
        for length, f in self._databases:
            if local_index < length:
                return local_index, f
            local_index -= length
        raise Exception(f"Trajectory index {global_index} was larger than total length {self.__len__()}")

    def __len__(self):
        # TODO implements this
        assert False

    def __getitem__(self, idx):
        # TODO implement this
        assert False

    @classmethod
    def get_dataloader(
            cls,
            directory,
            batch_size=8,
            num_workers=4,
            shuffle=True,
            acquire_test_dataset=False,
            mini=False,
            **kwargs,
    ):
        """

        Parameters
        ----------
        cls : This class itself
        directory : str
            Directory with dataset
        test : bool
            Whether this is a test dataset or train
        batch_size : int
        num_workers : int
        shuffle : bool

        Returns
        -------
        torch.DataLoader

        """
        if isinstance(directory, str):
            directory = Path(directory).expanduser().absolute()

        dataset = cls(
            directory,
            acquire_test_dataset=acquire_test_dataset,
            mini=mini,
            **kwargs,
        )
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )
