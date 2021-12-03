import os
import random

import numpy as np
import torch
from torch import distributed as dist


def set_random_seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def all_reduce(tensor, reduction='sum'):
    if dist.is_available():
        dist.all_reduce(tensor)
        if reduction == 'mean':
            tensor /= dist.get_world_size()
        dist.barrier()


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, device):
        self.device = device
        self.reset()

    def reset(self):
        self.sum = torch.tensor(0.0, device=self.device)
        self.count = torch.tensor(0, dtype=torch.long, device=self.device)

    def update(self, val, n=1):
        val = val.detach()
        self.sum += val * n
        self.count += n

    @property
    def average(self):
        return (self.sum / self.count).item()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


class Lighting:
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img: torch.Tensor):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


def add_inception_lighting(transform):
    lighting = Lighting(alphastd=0.1,
                        eigval=[0.2175, 0.0188, 0.0045],
                        eigvec=[[-0.5675, 0.7192, 0.4009],
                                [-0.5808, -0.0045, -0.8140],
                                [-0.5836, -0.6948, 0.4203]])
    transform.transforms.insert(-1, lighting)
    return transform
