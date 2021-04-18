
from torch import Tensor, einsum
import numpy as np
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union, cast
from model.Unet3D import *
from scipy.ndimage import distance_transform_edt as eucl_distance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Code inpired in: https://github.com/LIVIAETS/boundary-loss

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

# Distance maps
def one_hot2dist(seg: np.ndarray, resolution: Tuple[float, float, float] = None,
                 dtype=None) -> np.ndarray:

    assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[k] = eucl_distance(negmask, sampling=resolution) * negmask \
                - (eucl_distance(posmask, sampling=resolution) - 1) * posmask


    return res

def one_hot2hd_dist(seg: np.ndarray, resolution: Tuple[float, float, float] = None,
                    dtype=None) -> np.ndarray:
    """
    Used for https://arxiv.org/pdf/1904.10030.pdf,
    implementation from https://github.com/JunMa11/SegWithDistMap
    """
    # Relasx the assertion to allow computation live on only a
    # subset of the classes
    # assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(np.bool)

        if posmask.any():
            res[k] = eucl_distance(posmask, sampling=resolution)

    return res

def class2one_hot(seg: Tensor, K: int) -> Tensor:
    # Breaking change but otherwise can't deal with both 2d and 3d
    if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
         return class2one_hot(seg.unsqueeze(dim=0), K)[0]

    #assert sset(seg, list(range(K))), (uniq(seg), K)

    b,*img_shape = seg.shape
    res = torch.zeros((b, K, *img_shape), dtype=torch.int32).to(device).scatter_(1, seg[:, None, ...], 1)

    assert res.shape == (b, K, *img_shape)
    assert one_hot(res)

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, K, *_ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), K)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res

def probs2class(probs: Tensor) -> Tensor:
    b, _, *img_shape = probs.shape
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, *img_shape)

    return res

def distance_map(target):
    """
    Compute distance map.

    Input:
    : target (torch tensor): model output tensor
    """
    resized_gt = target.squeeze(dim=0).squeeze(dim=0)
    one_hot_gt = class2one_hot(torch.tensor(resized_gt[None, ...], dtype=torch.int64), K=2)[0]
    dist = one_hot2dist(one_hot_gt.clone().detach().cpu().numpy(),dtype=np.float32)
    dist = torch.tensor(dist[None,...],dtype=torch.float32).to(device)

    return dist

#  ================== Losses =============================
class BoundaryLoss():
    """
    Boundary Loss
    """
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]


    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(target)

        #Compute Distance Map
        dist_maps = distance_map(target)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        # Compute Boundary Loss
        multipled = einsum("bkxyz,bkxyz->bkxyz", pc, dc)

        loss = multipled.mean()

        return loss

class HausdorffLoss():
    """
    Hausdorff loss
    """
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs)
        assert probs.shape == target.shape

        B, K, *xyz = probs.shape  # type: ignore

        pc = cast(Tensor, probs[:, self.idc, ...].type(torch.float32))
        tc = cast(Tensor, target[:, self.idc, ...].type(torch.float32))
        assert pc.shape == tc.shape == (B, len(self.idc), *xyz)

        target_dm_npy: np.ndarray = np.stack([one_hot2hd_dist(tc[b].cpu().detach().numpy())
                                              for b in range(B)], axis=0)
        assert target_dm_npy.shape == tc.shape == pc.shape
        tdm: Tensor = torch.tensor(target_dm_npy, device=probs.device, dtype=torch.float32)

        pred_segmentation: Tensor = probs2one_hot(probs).cpu().detach()
        pred_dm_npy: np.nparray = np.stack([one_hot2hd_dist(pred_segmentation[b, self.idc, ...].numpy())
                                            for b in range(B)], axis=0)
        assert pred_dm_npy.shape == tc.shape == pc.shape
        pdm: Tensor = torch.tensor(pred_dm_npy, device=probs.device, dtype=torch.float32)

        delta = (pc - tc)**2
        dtm = tdm**2 + pdm**2

        multipled = einsum("bkwhs,bkwhs->bkwhs", delta, dtm)

        loss = multipled.mean()

        return loss

