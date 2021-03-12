


import torch
from torch import Tensor, einsum
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from utils.readers import *
import numpy as np
from scipy.ndimage import distance_transform_edt as eucl_distance
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union, cast

from dataset import BalancedPatchGenerator


class average_metrics(object):
    """
    Average metrics class, use the update to add the current metric and self.avg to get the avg of the metric
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())

def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def simplex(t: Tensor, axis=1) -> bool:
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


def one_hot2dist(seg: np.ndarray, resolution: Tuple[float, float, float] = None,
                 dtype=None) -> np.ndarray:
    #assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[k] = eucl_distance(negmask, sampling=resolution) * negmask \
                - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel

    return res


class BoundaryLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        #print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        #assert simplex(probs)
        #assert not one_hot(dist_maps)

       # pc = probs[:, self.idc, ...].type(torch.float32)
       # dc = dist_maps[:, self.idc, ...].type(torch.float32)

        pc = probs[:,0,:,:,:]
        dc = probs[:,0,:,:,:]

        multipled = einsum("bkwh,bkwh->bkwh", pc, dc)

        loss = multipled.mean()

        return loss


# boundary_loss = BoundaryLoss(idc=[0])
#
# for i, data in enumerate(loader):
#     x, y = data
#     dist = one_hot2dist(y.detach().numpy())
#     dist = torch.Tensor(dist)
#     plt.imshow(dist[0,0,:,:,20],cmap='gray')
#     plt.show()
#     bl_loss = boundary_loss(y, dist)
#     print(bl_loss.item())

