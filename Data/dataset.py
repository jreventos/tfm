from typing import Callable, Optional, Sequence, Union, Tuple, Any
import hashlib
import json
import sys
import threading

import numpy as np
import torch
from torch.utils.data import Dataset as _torchDataset

class Dataset(_torchDataset)
    """
    Class that reads the MRI slices images and the segmentation slices. 
    """

    def __init__(self, data: Sequence, transform: Optional[Callable] = None): # Optional[...] is a shorthand notation for Union[..., None], telling the type checker that either an object of the specific type is required, or None is required.
        """

        :param data: input data to load and transform the dataset
        :param transform: a callable data transform on input data
        """

        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        data = self.data[index]
        if self.transform is not None:
            data = apply_transform(self.transform, data)

        return data

def apply_transform(transform: Callable, data: object, map_items: bool = True):
    """
    Transform `data` with `transform`.
    If `data` is a list or tuple and `map_data` is True, each item of `data` will be transformed
    and this method returns a list of outcomes.
    otherwise transform will be applied once with `data` as the argument.
    Args:
        transform: a callable to be used to transform `data`
        data: an object to be transformed.
        map_items: whether to apply transform to each item in `data`,
            if `data` is a list or tuple. Defaults to True.
    Raises:
        with_traceback: applying transform {transform}.
    """
    try:
        if isinstance(data, (list, tuple)) and map_items:
            return [transform(item) for item in data]
        return transform(data)
    except Exception as e:
        raise type(e)(f"applying transform {transform}.").with_traceback(e.__traceback__)


