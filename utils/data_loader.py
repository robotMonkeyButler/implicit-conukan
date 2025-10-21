import os
import cv2
import PIL
import torch
import random
import numpy as np
import torchvision
import torchvision.transforms.functional as TF
import scipy.ndimage
from typing import Tuple
import albumentations as A

from torch import Tensor

from .augmentations import ElasticTransformations, RandomRotationWithMask

from albumentations.pytorch import ToTensorV2


cv2.setNumThreads(0)
random.seed(0)
torch.manual_seed(0)

class PolypDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            inp_path: str,
            gt_path: str,
            train_set_idx: list,
            val_set_idx: list,
            train: bool,
            transforms: torchvision.transforms,
            normalise: torchvision.transforms,
    ) -> None:
        super().__init__()
        self.train = train
        self.transforms = transforms
        self.normalise = normalise
        self.img_tensor, self.lbl_tensor = self.load_data(
            inp_path=inp_path, gt_path=gt_path
        )
        (
            self.trainset,
            self.valset,
            self.trainset_lbl,
            self.valset_lbl,
        ) = self.train_val_split(train_set_idx=train_set_idx, val_set_idx=val_set_idx)
        if train:
            self.data = self.trainset
            self.lbl = self.trainset_lbl
        else:
            self.data = self.valset
            self.lbl = self.valset_lbl

    def random_crop(
            self, img: torch.Tensor, lbl: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_start = random.randint(0, img.shape[-2] - 224)
        y_start = random.randint(0, img.shape[-1] - 224)

        x_end = x_start + 224
        y_end = y_start + 224

        img_cropped, lbl_cropped = (
            img[:, x_start:x_end, y_start:y_end],
            lbl[:, x_start:x_end, y_start:y_end],
        )

        return img_cropped, lbl_cropped

    # Crop centre region for validation and testing
    def centre_crop(
            self, img: torch.Tensor, lbl: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = img.shape[-2]
        y = img.shape[-1]
        img_cropped, lbl_cropped = (
            img[:, x // 2 - 128: x // 2 + 128, y // 2 - 128: y // 2 + 128],
            lbl[:, x // 2 - 128: x // 2 + 128, y // 2 - 128: y // 2 + 128],
        )
        return img_cropped, lbl_cropped

    def load_data(self, inp_path: str, gt_path: str) -> torch.Tensor:
        img_arr = []
        lbl_arr = []

        assert os.path.isdir(inp_path) and os.path.isdir(gt_path)
        for img_name in os.listdir(inp_path):
            img_path = os.path.join(inp_path, img_name)
            lbl_path = os.path.join(gt_path, img_name)
            if os.path.isfile(img_path) and os.path.isfile(lbl_path):
                img = cv2.imread(img_path)
                img = np.transpose(np.array(img / 255, dtype=np.float32), (2, 0, 1))
                img = torch.tensor(img)
                img = TF.resize(img=img, size=(256, 256))
                img_cp = torch.zeros_like(img)
                for i in range(2):
                    img_cp[i] = img[2 - i]

                lbl = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
                lbl = np.array(lbl, dtype=np.float32)[np.newaxis, :, :]
                lbl_new = (lbl >= 127.5).astype(np.float16)

                lbl_new = torch.tensor(lbl_new)
                lbl_new = TF.resize(img=lbl_new, size=(256, 256), interpolation=TF.InterpolationMode.NEAREST)
                # crop to 224x224
                # if self.train:
                #    img_cp, lbl_new = self.random_crop(img=img_cp, lbl=lbl_new)
                img_arr.append(img_cp)
                lbl_arr.append(lbl_new)
        img_tensor = torch.stack(img_arr)
        lbl_tensor = torch.stack(lbl_arr)

        return img_tensor, lbl_tensor

    def train_val_split(
            self, train_set_idx: list, val_set_idx: list
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        train_img_arr = []
        train_lbl_arr = []
        val_img_arr = []
        val_lbl_arr = []

        for i in train_set_idx:
            train_img_arr.append(self.img_tensor[i].numpy())
            train_lbl_arr.append(self.lbl_tensor[i].numpy())

        for j in val_set_idx:
            val_img_arr.append(self.img_tensor[j].numpy())
            val_lbl_arr.append(self.lbl_tensor[j].numpy())

        trainset, valset = torch.tensor(
            np.array(train_img_arr, dtype=np.float32)
        ), torch.tensor(np.array(val_img_arr, dtype=np.float32))
        trainset_lbl, valset_lbl = torch.tensor(
            np.array(train_lbl_arr, dtype=np.float32)
        ), torch.tensor(np.array(val_lbl_arr, dtype=np.float32))
        return trainset, valset, trainset_lbl, valset_lbl

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = {"data": self.data[idx], "lbl": self.lbl[idx]}

        if self.transforms is not None:
            sample = self.transforms(sample)
            sample['lbl'] = (sample['lbl'] >= 0.5).to(torch.float)
        if self.normalise is not None:
            sample['data'] = self.normalise(sample['data'])

        return sample["data"], sample["lbl"]

    def __len__(self) -> int:
        return len(self.data)


class NucleiDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            inp_path: str,
            train_set_idx: list,
            val_set_idx: list,
            train: bool,
            transforms: torchvision.transforms,
            normalise: torchvision.transforms,
    ) -> None:
        super().__init__()
        self.transforms = transforms
        self.normalise = normalise

        self.img_tensor, self.lbl_tensor = self.load_data(inp_path=inp_path)

        (
            self.trainset,
            self.valset,
            self.trainset_lbl,
            self.valset_lbl,
        ) = self.train_val_split(train_set_idx=train_set_idx, val_set_idx=val_set_idx)

        if train:
            self.data = self.trainset
            self.lbl = self.trainset_lbl
        else:
            self.data = self.valset
            self.lbl = self.valset_lbl

    def load_data(self, inp_path: str) -> tuple[Tensor, Tensor]:
        img_arr = []
        lbl_arr = []

        assert os.path.isdir(inp_path)
        for img_name in os.listdir(inp_path):
            img_path = os.path.join(inp_path, img_name, "images", img_name + ".png")
            lbl_path = os.path.join(inp_path, img_name, "masks")
            if os.path.isfile(img_path):
                img = cv2.imread(img_path)
                img = cv2.resize(img, (256, 256))
                img = np.transpose(
                    np.array(img / 255, dtype=np.float32), (2, 0, 1)
                )
                img = torch.tensor(img)
                img_arr.append(img)  # changed to range [0,1]
            mask_full = torch.zeros((1, 256, 256))
            if os.path.isdir(lbl_path):
                for f in os.listdir(lbl_path):
                    lbl_img_path = os.path.join(lbl_path, f)
                    lbl = cv2.imread(lbl_img_path, cv2.IMREAD_GRAYSCALE)
                    lbl = np.array(lbl, dtype=np.float32)[np.newaxis, :, :]
                    lbl_new = (lbl >= 127.5).astype(np.float16)
                    lbl_new = torch.tensor(lbl_new)
                    lbl_new = TF.resize(img=lbl_new, size=[256, 256], interpolation=TF.InterpolationMode.NEAREST)
                    mask_full = torch.maximum(mask_full, lbl_new)
                lbl_arr.append(mask_full)
        img_tensor = torch.stack(img_arr)
        lbl_tensor = torch.stack(lbl_arr)

        return img_tensor, lbl_tensor

    def train_val_split(
            self, train_set_idx: list, val_set_idx: list
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        train_img_arr = []
        train_lbl_arr = []
        val_img_arr = []
        val_lbl_arr = []

        for i in train_set_idx:
            train_img_arr.append(self.img_tensor[i].numpy())
            train_lbl_arr.append(self.lbl_tensor[i].numpy())

        for j in val_set_idx:
            val_img_arr.append(self.img_tensor[j].numpy())
            val_lbl_arr.append(self.lbl_tensor[j].numpy())

        trainset, valset = torch.tensor(
            np.array(train_img_arr, dtype=np.float32)
        ), torch.tensor(np.array(val_img_arr, dtype=np.float32))
        trainset_lbl, valset_lbl = torch.tensor(
            np.array(train_lbl_arr, dtype=np.float32)
        ), torch.tensor(np.array(val_lbl_arr, dtype=np.float32))
        return trainset, valset, trainset_lbl, valset_lbl

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = {"data": self.data[idx], "lbl": self.lbl[idx]}
        if self.normalise is not None:
            sample['data'] = self.normalise(sample['data'])
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample["data"], sample["lbl"]

    def __len__(self) -> int:
        return len(self.data)


class BreastCancerDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            inp_path: str,
            train_set_idx: list,
            val_set_idx: list,
            train: bool,
            transforms: torchvision.transforms,
            normalise: torchvision.transforms
    ) -> None:
        super().__init__()
        self.train = train
        self.transforms = transforms
        self.normalise = normalise
        self.img_tensor, self.lbl_tensor = self.load_data(inp_path=inp_path)
        (
            self.trainset,
            self.valset,
            self.trainset_lbl,
            self.valset_lbl,
        ) = self.train_val_split(train_set_idx=train_set_idx, val_set_idx=val_set_idx)
        if train:
            self.data = self.trainset
            self.lbl = self.trainset_lbl
        else:
            self.data = self.valset
            self.lbl = self.valset_lbl

    def random_crop(
            self, img: torch.Tensor, lbl: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_start = random.randint(0, img.shape[-2] - 224)
        y_start = random.randint(0, img.shape[-1] - 224)

        x_end = x_start + 224
        y_end = y_start + 224

        img_cropped, lbl_cropped = (
            img[:, x_start:x_end, y_start:y_end],
            lbl[:, x_start:x_end, y_start:y_end],
        )

        return img_cropped, lbl_cropped

    def load_data(self, inp_path: str) -> tuple[Tensor, Tensor]:
        img_arr = []
        lbl_arr = []

        assert os.path.isdir(inp_path)
        count_invalid = 0

        for subdir in os.listdir(inp_path):

            if subdir != "normal" and subdir != "benign_test" and subdir != "malignant_test":
                # if subdir != "normal" and subdir != "benign" and subdir != "malignant":
                subdir = os.path.join(inp_path, subdir)
                for img_name in os.listdir(subdir):
                    if "mask" not in img_name:
                        img_path = os.path.join(subdir, img_name)
                        lbl_path = os.path.join(
                            subdir, img_name[:-4] + "_mask.png"
                        )
                        lbl_path_add_mask1 = os.path.join(
                            subdir, img_name[:-4] + "_mask_1.png"
                        )
                        lbl_path_add_mask2 = os.path.join(
                            subdir, img_name[:-4] + "_mask_2.png"
                        )

                        if os.path.isfile(img_path):
                            # print(img_path)
                            # print("===================")
                            img = cv2.imread(img_path)
                            if img is None:
                                continue
                            img = cv2.resize(img, (256, 256))
                            img = np.transpose(
                                np.array(img / 255, dtype=np.float32), (2, 0, 1)
                            )
                            img = torch.tensor(img)
                            img_arr.append(img)  # changed to range [0,1]

                        mask_full = torch.zeros((1, 256, 256))
                        if os.path.isfile(lbl_path):
                            lbl_img_path = os.path.join(lbl_path)
                            lbl = cv2.imread(lbl_img_path, cv2.IMREAD_GRAYSCALE)
                            lbl = np.array(lbl, dtype=np.float32)[np.newaxis, :, :]
                            lbl_new = (lbl >= 127.5).astype(np.float16)
                            lbl_new = torch.tensor(lbl_new)
                            lbl_new = TF.resize(img=lbl_new, size=[256, 256],
                                                interpolation=TF.InterpolationMode.NEAREST)
                            mask_full = torch.maximum(mask_full, lbl_new)

                        if os.path.isfile(lbl_path_add_mask1):
                            lbl_img_path = os.path.join(lbl_path_add_mask1)
                            lbl = cv2.imread(lbl_img_path, cv2.IMREAD_GRAYSCALE)
                            lbl = np.array(lbl, dtype=np.float32)[np.newaxis, :, :]
                            lbl_new = (lbl >= 127.5).astype(np.float16)
                            lbl_new = torch.tensor(lbl_new)
                            lbl_new = TF.resize(img=lbl_new, size=[256, 256],
                                                interpolation=TF.InterpolationMode.NEAREST)
                            mask_full = torch.maximum(mask_full, lbl_new)

                        if os.path.isfile(lbl_path_add_mask2):
                            lbl_img_path = os.path.join(lbl_path_add_mask2)
                            lbl = cv2.imread(lbl_img_path, cv2.IMREAD_GRAYSCALE)
                            lbl = np.array(lbl, dtype=np.float32)[np.newaxis, :, :]
                            lbl_new = (lbl >= 127.5).astype(np.float16)
                            lbl_new = torch.tensor(lbl_new)
                            lbl_new = TF.resize(img=lbl_new, size=[256, 256],
                                                interpolation=TF.InterpolationMode.NEAREST)
                            mask_full = torch.maximum(mask_full, lbl_new)

                        lbl_arr.append(mask_full)
                    else:
                        pass
        img_tensor = torch.stack(img_arr)
        lbl_tensor = torch.stack(lbl_arr)

        return img_tensor, lbl_tensor

    def train_val_split(
            self, train_set_idx: list, val_set_idx: list
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        train_img_arr = []
        train_lbl_arr = []
        val_img_arr = []
        val_lbl_arr = []

        for i in train_set_idx:
            train_img_arr.append(self.img_tensor[i])
            train_lbl_arr.append(self.lbl_tensor[i])

        for j in val_set_idx:
            val_img_arr.append(self.img_tensor[j])
            val_lbl_arr.append(self.lbl_tensor[j])

        trainset, valset = torch.stack(train_img_arr), torch.stack(val_img_arr)
        trainset_lbl, valset_lbl = torch.stack(train_lbl_arr), torch.stack(val_lbl_arr)

        return trainset, valset, trainset_lbl, valset_lbl

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = {"data": self.data[idx], "lbl": self.lbl[idx]}
        if self.normalise is not None:
            sample['data'] = self.normalise(sample['data'])
        # if self.train:
        #    sample['data'], sample['lbl'] = self.random_crop(img=sample['data'], lbl=sample['lbl'])
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample["data"], sample["lbl"]

    def __len__(self) -> int:
        return len(self.data)


class MelanomaDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            inp_path: str,
            gt_path: str,
            train_set_idx: list,
            val_set_idx: list,
            train: bool,
            transforms: torchvision.transforms,
            normalise: torchvision.transforms,
            test: bool = False,
    ) -> None:
        super().__init__()
        self.transforms = transforms
        self.normalise = normalise
        self.img_tensor, self.lbl_tensor = self.load_data(
            inp_path=inp_path, gt_path=gt_path
        )
        if not test:
            (
                self.trainset,
                self.valset,
                self.trainset_lbl,
                self.valset_lbl,
            ) = self.train_val_split(train_set_idx=train_set_idx, val_set_idx=val_set_idx)
            if train:
                self.data = self.trainset
                self.lbl = self.trainset_lbl
            else:
                self.data = self.valset
                self.lbl = self.valset_lbl
        else:
            self.data = self.img_tensor
            self.lbl = self.lbl_tensor

    def load_data(self, inp_path: str, gt_path: str) -> tuple[Tensor, Tensor]:
        img_arr = []
        lbl_arr = []

        assert os.path.isdir(inp_path) and os.path.isdir(gt_path)
        for img_name in os.listdir(inp_path):
            img_path = os.path.join(inp_path, img_name)
            lbl_path = os.path.join(gt_path, img_name[:-4] + "_Segmentation.png")
            if os.path.isfile(img_path):
                img = cv2.imread(img_path)
                img = cv2.resize(img, (512, 512))
                img = np.transpose(np.array(img / 255, dtype=np.float32), (2, 0, 1))
                img = torch.tensor(img)
                img_arr.append(img)

            if os.path.isfile(lbl_path):
                lbl = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
                lbl = np.array(lbl, dtype=np.float32)[np.newaxis, :, :]
                lbl_new = (lbl >= 127.5).astype(np.float16)
                lbl_new = torch.tensor(lbl_new)
                lbl_new = TF.resize(img=lbl_new, size=[512, 512], interpolation=TF.InterpolationMode.NEAREST)
                lbl_arr.append(lbl_new)
        img_tensor = torch.stack(img_arr)
        lbl_tensor = torch.stack(lbl_arr)

        return img_tensor, lbl_tensor

    def train_val_split(
            self, train_set_idx: list, val_set_idx: list
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        train_img_arr = []
        train_lbl_arr = []
        val_img_arr = []
        val_lbl_arr = []

        for i in train_set_idx:
            train_img_arr.append(self.img_tensor[i].numpy())
            train_lbl_arr.append(self.lbl_tensor[i].numpy())

        for j in val_set_idx:
            val_img_arr.append(self.img_tensor[j].numpy())
            val_lbl_arr.append(self.lbl_tensor[j].numpy())

        trainset, valset = torch.tensor(
            np.array(train_img_arr, dtype=np.float32)
        ), torch.tensor(np.array(val_img_arr, dtype=np.float32))
        trainset_lbl, valset_lbl = torch.tensor(
            np.array(train_lbl_arr, dtype=np.float32)
        ), torch.tensor(np.array(val_lbl_arr, dtype=np.float32))
        return trainset, valset, trainset_lbl, valset_lbl

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = {"data": self.data[idx], "lbl": self.lbl[idx]}
        if self.normalise is not None:
            sample['data'] = self.normalise(sample['data'])
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample["data"], sample["lbl"]

    def __len__(self) -> int:
        return len(self.data)
