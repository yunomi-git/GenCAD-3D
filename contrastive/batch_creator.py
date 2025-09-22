from abc import ABC, abstractmethod
from geometry.geometry_data import GeometryLoader
import numpy as np
from cadlib.macro import AVAILABLE_SEQUENCE_LENGTHS
from utils.util import Stopwatch
import random

class BatchCreator(ABC):
    @abstractmethod
    def get_batch(self, num_samples, randomize):
        pass


class RandomBatchCreator(BatchCreator):
    def __init__(self, geometry_loader:GeometryLoader, encoder_type):
        self.geometry_loader = geometry_loader
        self.num_in_space = None

        # if "mesh" in encoder_type:
        #     self.all_cad_ids = self.geometry_loader.get_all_valid_data(cad=True, mesh=True)
        # else:
        #     self.all_cad_ids = self.geometry_loader.get_all_valid_data(cad=True, cloud=True)
        self.all_cad_ids = self.geometry_loader.get_all_valid_data(**self.geometry_loader.standard_modality_checks(encoder_type))
        self.num_in_space = len(self.all_cad_ids)



    def get_batch(self, num_samples, randomize):
        if randomize:
            self.subbatch_indices = np.random.choice(np.arange(self.num_in_space), size=num_samples, replace=False)
        else:
            self.subbatch_indices = np.arange(num_samples)

        return [self.all_cad_ids[idx] for idx in self.subbatch_indices]



class BalancedBatchCreator(BatchCreator):
    def __init__(self, geometry_loader:GeometryLoader, num_bins, encoder_type, uniform_subbatch_sizes=True):
        # find limits of each batch
        sequence_limits = AVAILABLE_SEQUENCE_LENGTHS
        self.num_seq_lengths = len(sequence_limits)
        self.sl_bounds = {i: (min(sequence_limits) + i * self.num_seq_lengths // num_bins,
                              min(sequence_limits) + (i+1) * self.num_seq_lengths // num_bins - 1) for i in range(num_bins)}
        self.sl_bounds[num_bins - 1] = (min(sequence_limits) + (num_bins - 1) * self.num_seq_lengths // num_bins,
                                        max(sequence_limits))

        self.uniform_subbatch_size = uniform_subbatch_sizes

        # for each batch make a dictionary
        self.sl_batch_ids = {i: geometry_loader.get_cad_ids_of_sequence_length(min_length=self.sl_bounds[i][0],
                                                                               max_length=self.sl_bounds[i][1],
                                                                               **geometry_loader.standard_modality_checks(encoder_type))
                             for i in range(num_bins)}

        self.num_batches = num_bins

        self.sl_num_items_per_batch = {i: len(self.sl_batch_ids[i]) for i in range(num_bins)}

    def get_batch(self, num_samples, randomize):
        batch = []

        #
        if self.uniform_subbatch_size:
            sl_num_to_sample = np.ones(self.num_batches)
        else:
            sl_num_to_sample = np.random.rand(self.num_batches)
        sl_num_to_sample /= np.sum(sl_num_to_sample)
        sl_num_to_sample = (num_samples * sl_num_to_sample).astype(np.int32)
        sl_num_to_sample[self.num_batches - 1] = num_samples - np.sum(sl_num_to_sample[:self.num_batches - 1])

        # print(sl_num_to_sample)

        for i in range(self.num_batches):
            num_ids_in_subbatch = len(self.sl_batch_ids[i])
            # num_items_to_sample = num_items_per_sample
            # if i == self.num_batches - 1:
            #     num_items_to_sample = last_num_items_in_sample
            try:
                if randomize:
                    self.subbatch_indices = np.random.choice(np.arange(num_ids_in_subbatch), size=sl_num_to_sample[i], replace=False)
                else:
                    self.subbatch_indices = np.arange(sl_num_to_sample[i])
            except Exception as e:
                print(e)
                print("bounds", self.sl_bounds[i], "requested", sl_num_to_sample[i], "available", num_ids_in_subbatch)
            subbatch = [self.sl_batch_ids[i][idx] for idx in self.subbatch_indices]
            batch += subbatch

        random.shuffle(batch)
        return batch

if __name__=="__main__":
    watch = Stopwatch()
    data_root = "/mnt/barracuda/CachedDatasets/GenCAD3D/"
    geometry_loader = GeometryLoader(data_root, phase="test", geometry_subdir="meshes_rm2kfix/")
    batch_creator = BalancedBatchCreator(geometry_loader=geometry_loader, num_bins=6)
    # batch_creator = RandomBatchCreator(geometry_loader=geometry_loader)
    watch.start()
    for i in range(1000):
        batch = batch_creator.get_batch(5, randomize=True)
        print(batch)
    watch.print_time()
    print()