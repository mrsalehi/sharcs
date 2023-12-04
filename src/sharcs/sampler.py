from distutils.errors import DistutilsInternalError
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler
import torch
import math
import random
import numpy as np
from torch.utils.data import BatchSampler
import logging
import sys
import os
import json
from collections import defaultdict
from utils import create_binary_width_label
from collections import Counter
from loguru import logger


class SingleWidthBatchSampler(DistributedSampler):
    def __init__(self, dataset, batch_size: int, num_buckets: int, args, num_replicas=None,
                 rank=None, shuffle=True, seed: int = 0, drop_last=False) -> None:
        """
        drop_last for the super class would be False but after intializing the super I am going to use drop_last True for each bucket in pre_epoch_setup
        """
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.num_buckets = num_buckets
        self.batch_size = batch_size
        assert args.use_router
        self.args = args
        
    def set_buckets_and_batches(self, epoch, config):
        """
        Assign the buckets based on the training history and create batches of data based on that
        train_history: this will be used in the online case which is the confidence scores of the last k epochs where k in the window size.
        """
        self.set_epoch(epoch) 
        
        # 1. CREATING LABELS FOR TRAINING THE ROUTER WHICH IS self.index_to_width_mult_label
        bucket_fpath = os.path.join(self.args.result_dir, f'online_buckets_for_ep{epoch}.json')
        index_to_width_mult_fpath = os.path.join(self.args.result_dir, f'uid_to_width_mults_for_ep{epoch}.json')

        if os.path.exists(bucket_fpath): 
            # NOTE:
            # 1. The labels for training the router will be in self.index_to_width_mult_label
            # which is created from index_to_width_mult. Look at save_and_udpate_train_history 
            # in utils to see how those labels are created.
            # 2. Buckets for this epoch will be in self.buckets. Note that each sample will be processed by
            # just one width in each epoch even if it has multiple width labels assigned to it.
            # The router is still trained with both of the width labels. In this way things get much 
            # more simpler than doing full forward pass for task classification for all of the 
            # assigned widths...
            logger.info(f"Epoch {epoch}: Loading buckets from {bucket_fpath}.")
            self.buckets = json.load(open(bucket_fpath))
            self.buckets = {float(k): v for k, v in self.buckets.items()}
            index_to_width_mult = json.load(open(index_to_width_mult_fpath))
            index_to_width_mult = {int(k): v for k, v in index_to_width_mult.items()}
            # convert keys to float
            logger.info(f"Length of buckets in the beginning of epoch {epoch}: {[len(v) for v in self.buckets.values()]}.")
        else:
            # during the first window_size epochs where it is just full width.
            self.buckets = {width_: [] for width_ in self.args.width_mult_list}
            biggest_width = self.args.width_mult_list[-1]
            # self.buckets[1.0] = self.dataset.tensors[0].numpy().tolist()
            self.buckets[biggest_width] = self.dataset.tensors[0].numpy().tolist()
            # index_to_width_mult = {i: [1.0] for i in range(len(self.dataset.tensors[0]))}
            index_to_width_mult = {i: [biggest_width] for i in range(len(self.dataset.tensors[0]))}

        self.bucket_lengths_before_repeat_or_aug = {width_: len(indices) for width_, indices in self.buckets.items()}  # will be used to track number of unique samples in each bucket 

        self.index_to_width_mult_label = {i: create_binary_width_label(self.args.width_mult_list, width_mults) \
            for i, width_mults in index_to_width_mult.items()}

        g = torch.Generator() 

        # 2. REPEATING THE SMALLER SIZED BUCKETS OR ONLINE AUGMENTATION
        if config.use_router:
            if config.repeat_smaller_sized_buckets:
                # Repeating the training samples for the sub-networks with less training samples
                # till the size of the buckets are roughly equal.
                logger.info("Repeating smaller sized buckets policy.")
                max_bucket_size = max([len(bucket) for bucket in self.buckets.values() if len(bucket) > 0])
                logger.info(f"Maximum size of buckets is {max_bucket_size}.")

                for bucket_width, bucket in self.buckets.items():
                    if 0 < len(bucket) < max_bucket_size:
                        bucket = bucket * (max_bucket_size // len(bucket)) 
                        if len(bucket) < max_bucket_size:
                            padding = max_bucket_size - len(bucket)
                            self.buckets[bucket_width] = bucket + bucket[:padding]
                        logger.info(f"Length of bucket {bucket_width} is {len(bucket)}. Repeating the data in the bucket {(max_bucket_size // len(bucket))} times...")
            else:
                logger.info("Not repeating the samples for the sub-networks with less training samples.")
                # either repeating samples or doing online augmentation
        else:
            raise NotImplementedError


        if self.args.verbose:
            logger.info(f"rank: {self.rank}, buckets: {self.buckets}")
        
        # shuffle the elements in each bucket. This should be the same for all ranks.
        g.manual_seed(self.seed + self.epoch + 100)
        for bucket_width, bucket in self.buckets.items():
            indices = torch.randperm(len(bucket), generator=g).tolist()
            self.buckets[bucket_width] = [bucket[idx] for idx in indices]
        
        # 3. CREATING THE BATCHES OF INDIVIDUAL BUCKETS
        batches = [] 

        for bucket in self.buckets.values():
            bucket_batches = [bucket[i:i+self.batch_size] for i in range(0, len(bucket), self.batch_size)]
            if bucket_batches:
                # making sure that all of the batches have the same number of elements
                if len(bucket_batches[-1]) < self.batch_size:
                    # borrow some elements from the first batch
                    batch_to_borrow_from = bucket_batches[0]
                    padding_size = self.batch_size - len(bucket_batches[-1])
                    bucket_batches[-1] += batch_to_borrow_from[:padding_size]

            batches.append(bucket_batches)
        
        # batches is a list of lists now: [[bucket1_batch1, bucket1_batch2, ...], [bucket2_batch1, bucket2_batch2, ...], ... [bucketN_batch1, bucketN_batch2, ...]]

        # 4. CREATING THE BATCHES OF ALL THE BUCKETS FOR TRAINING
        if self.epoch < self.args.train_router_window_size:
            logger.info(f"Still in the first phase of training. Just taking the batches for training the largest subnetwork with width {self.args.width_mult_list[-1]}.")
            self.batches = []
            for batch in batches[-1]:
                self.batches.append([{
                    "index": b,
                    "width_mult_label": self.index_to_width_mult_label[b],
                    "batch_width": len(batches) - 1,  # index of the last or biggest bucket (most likely there are three widths and the index of the last bucket is 2)
                    "per_iteration_bucket_loss_weight": 1.0
                } \
                    for b in batch])
        else: 
            if self.args.repeat_smaller_sized_buckets:
                assert all([len(l) > 0 for l in batches])
                self.batches = []
                logger.info(f"buckets have {[len(b) for b in batches]} number of batches.")
                for bucket_batches in zip(*batches):
                    for bucket_idx, batch in enumerate(bucket_batches):
                        self.batches.append([{
                            "index": b,
                            "width_mult_label": self.index_to_width_mult_label[b],
                            "batch_width": bucket_idx,  # index of the last or biggest bucket (most likely there are three widths and the index of the last bucket is 2)
                            "per_iteration_bucket_loss_weight": 1.0
                        } \
                            for b in batch])
                     
                logger.info(f"In total will process {len(self.batches)} batches of data in this epoch.")
            else:
                # not going to repeat the smaller buckets
                # NOTE: this is a stupid way but it works. I am going to append None to the end of batches of smaller buckets
                # until they have the same length as the largest bucket.
                max_len = max([len(bucket_batches) for bucket_batches in batches])
                for i, batch in enumerate(batches):
                    if len(batch) < max_len:
                        batches[i] += [None] * (max_len - len(batch))
                
                self.batches = []
                for bucket_batches in zip(*batches):
                    for bucket_idx, batch in enumerate(bucket_batches):
                        if batch is not None:
                            self.batches.append([{
                                "index": b,
                                "width_mult_label": self.index_to_width_mult_label[b],
                                "batch_width": bucket_idx,  # index of the last or biggest bucket (most likely there are three widths and the index of the last bucket is 2)
                                "per_iteration_bucket_loss_weight": 1.0
                            } \
                                for b in batch])
  
        if self.drop_last and len(self.batches) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.batches) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(self.batches) / self.num_replicas)
        
        self.total_size = self.num_samples * self.num_replicas
 
    def __iter__(self):
        """
        returns the batch + adaptive width multiplier
        adaptive_width will be used as some sort of label in training of the model
        """
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            # g = torch.Generator()
            # g.manual_seed(self.seed + self.epoch)
            # indices = torch.randperm(len(self.batches), generator=g).tolist()
            # batches = [self.batches[idx] for idx in indices]
            # for now disabling the shuffling here for two reasons:
            # 1. The elements for each width multiplier are well shuffled before batching.
            # 2. self.batches alternate between different widths (i.e. 0.25, 0.5, 1.0, 0.25, 0.5, ...) and shuffling them will mess up the order.
            batches = self.batches 
        else:
            indices = list(range(len(self.batches)))
            batches = batches[indices]

        if not self.drop_last:
            # add extra batches to make it evenly divisible by the number of gpus
            padding_size = self.total_size - len(batches)
            if padding_size <= len(batches):
                batches += batches[:padding_size]
            else:
                batches += (batches * math.ceil(padding_size / len(batches)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            batches = batches[:self.total_size]
        assert len(batches) == self.total_size
        
        batches = batches[self.rank:self.total_size:self.num_replicas]
        if self.args.enable_per_iteration_bucket_loss_weighting: # change the weight from default 1. to values based on ratio of buckets
            for i in range(0, len(batches), self.num_replicas):
                iteration_batches = batches[i:i+self.num_replicas] 
                iteration_batch_widths = [batch[0]["batch_width"] for batch in iteration_batches]  # note that this is not the exact width but it is the bucket index
                counter = Counter(iteration_batch_widths)
                sum_values = sum(counter.values())
                counter = {k: sum_values / v for k, v in counter.items()} 
                for batch in iteration_batches:
                    for b in batch:
                        b["per_iteration_bucket_loss_weight"] = counter[b["batch_width"]]

        if self.args.verbose:
            logger.info(f"rank: {self.rank}, batches: {batches}")
        
        assert len(batches) == self.num_samples

        return iter(batches)

   
    def __len__(self):
        return self.num_samples