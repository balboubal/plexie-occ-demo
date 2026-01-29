# Minimal util/misc.py for P2PNet inference
# Based on DETR's misc.py

import torch
import torch.distributed as dist
from typing import Optional, List

class NestedTensor(object):
    """Simple wrapper around a tensor with optional mask."""
    def __init__(self, tensors, mask: Optional[torch.Tensor] = None):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[torch.Tensor]):
    """Create a NestedTensor from a list of tensors (for batching)."""
    # For simplicity, just stack tensors (assuming same size)
    if len(tensor_list) == 1:
        return tensor_list[0]
    return torch.stack(tensor_list)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k."""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def is_dist_avail_and_initialized():
    """Check if distributed training is available and initialized."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """Get world size for distributed training."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    """Wrapper around F.interpolate."""
    return torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners)
