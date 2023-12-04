import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset
from collections.abc import Mapping
from dataclasses import dataclass


class OnlineBucketsTensorDataset(TensorDataset):
    """
    dataset class for glue dataset (not squad)
    """
    def __init__(self, args, *tensors):
        super().__init__(*tensors)
        self.window_size = args.train_router_window_size
    
    def __getitem__(self, input_index):
        if isinstance(input_index, dict):
            # using SingleWidthBatchSampler 
            index = input_index['index']
            width_mult_label = input_index['width_mult_label']  # a list as we can have multiple width mults
            batch_width = input_index['batch_width']  # width of the network that should be used for this batch
            per_iteration_bucket_loss_weighting = input_index['per_iteration_bucket_loss_weight']
            unique_id, input_ids, attention_mask, token_type_ids, label = \
                tuple(tensor[index] for tensor in self.tensors)
            aug = input_index.get('aug', None)

            if aug:
                # means we are doing online augmentation
                input_ids = aug(input_ids)

            data = (unique_id, input_ids, attention_mask, token_type_ids, label, width_mult_label, batch_width, per_iteration_bucket_loss_weighting)
        else:
            # using a plain sampler when doing distillation on aug data
            unique_id, input_ids, attention_mask, token_type_ids, label = \
                tuple(tensor[input_index] for tensor in self.tensors)            
            data = (unique_id, input_ids, attention_mask, token_type_ids, label)

        return data


def collate_fn(batch):  
    """
    collate function for glue dataset (not squad)
    """
    unique_ids, input_ids, attention_masks, token_type_ids, labels, width_mult_label, batch_width, per_iteration_bucket_loss_weighting = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    unique_ids = torch.stack(unique_ids, dim=0)
    attention_masks = torch.stack(attention_masks, dim=0)
    token_type_ids = torch.stack(token_type_ids, dim=0)
    labels = torch.stack(labels, dim=0)
    width_mult_label = torch.stack(width_mult_label, dim=0)
    batch_width = torch.tensor(batch_width, dtype=torch.long)  # TODO: all of the values should be the same, so we should be able to use just the first one
    per_iteration_bucket_loss_weighting = torch.tensor(per_iteration_bucket_loss_weighting, dtype=torch.float) # same as above is a single value..

    return unique_ids, input_ids, attention_masks, token_type_ids, labels, width_mult_label, batch_width, per_iteration_bucket_loss_weighting
    
 
def torch_default_data_collator(features):
    import torch

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch
