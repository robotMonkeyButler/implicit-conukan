from typing import Dict
import torch
import numpy as np
import torchvision.transforms.functional as TF

torch.manual_seed(0)
np.random.seed(0)


class RandomRotation(object):
    def __init__(self) -> None:
        pass

    def __call__(self, sample: Dict) -> Dict:
        data, lbl = sample["data"], sample["lbl"]
        rnd = np.random.randint(0, 3)
        data = TF.rotate(img=data, angle=rnd * 90)
        lbl = TF.rotate(img=lbl, angle=rnd * 90)
        sample["data"] = data
        sample["lbl"] = lbl
        return sample


class RandomVerticalFlip(object):
    def __init__(self, p) -> None:
        self.p = p

    def __call__(self, sample: Dict) -> Dict:
        data, lbl = sample["data"], sample["lbl"]
        if torch.rand(1) < self.p:
            data = TF.vflip(data)
            lbl = TF.vflip(lbl)
        sample["data"] = data
        sample["lbl"] = lbl
        return sample


class RandomHorizontalFlip(object):
    def __init__(self, p) -> None:
        self.p = p

    def __call__(self, sample: Dict) -> Dict:
        data, lbl = sample["data"], sample["lbl"]
        if torch.rand(1) < self.p:
            data = TF.hflip(data)
            lbl = TF.hflip(lbl)
        sample["data"] = data
        sample["lbl"] = lbl
        return sample
