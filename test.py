import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torchmetrics
import torchvision
from typing import Tuple

import warnings

from utils.data_loader import (
    PolypDataset,
    BreastCancerDataset,
    MelanomaDataset,
    NucleiDataset,
)
import ml_collections
from implicit_conukan.archs_conmultukan import conMultUKAN
from utils.metrics import iou_score, hd95_score, pixelwise_acc, ObjectDice, F1score
from ptflops import get_model_complexity_info

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="compute metrics")

parser.add_argument(
    "--dataset",
    type=str,
    choices=["polyp", "nuclei", "breast_cancer", "melanoma_segmentation"],
    help="Dataset to use.",

)
parser.add_argument(
    "--net",
    nargs="+",
    type=str,
    help="Network architecture.",
)
parser.add_argument(
    "--identifier",
    nargs="+",
    type=str,
    help="Training identifier",
)
parser.add_argument(
    "--path",
    type=str,
    help="Path",
)
parser.add_argument(
    "--pretrained",
    type=bool,
    help="Use pretrained UNet model",
)
parser.add_argument("--solver", type=str, default="rk4", nargs='*', help="Solver")


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


def load_dataset(dataset: str) -> Tuple[torch.utils.data.Dataset, float]:
    if dataset == "polyp":
        data_path = (
            "../datasets/Kvasir-SEG/test"
        )
        inp_path = os.path.join(data_path, "images/test")
        gt_path = os.path.join(data_path, "masks/test")
        perm_idx = torch.randperm(120)
        val_set_idx = perm_idx[:24]
        train_set_idx = perm_idx[24:]
        valset = PolypDataset(
            inp_path=inp_path,
            gt_path=gt_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=False,
            transforms=None,
            normalise=None
        )
        img_size = 256
    elif dataset == "nuclei":
        data_path = (
            "../datasets/nuclei/test"
        )
        perm_idx = torch.randperm(133)
        val_set_idx = perm_idx[:27]
        train_set_idx = perm_idx[27:]
        valset = NucleiDataset(
            inp_path=data_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=False,
            transforms=None,
            normalise=None
        )
        img_size = 256

    elif dataset == "breast_cancer":
        data_path = "../datasets/breast_cancer"
        perm_idx = torch.randperm(195)
        val_set_idx = perm_idx[:39]
        train_set_idx = perm_idx[39:]
        valset = BreastCancerDataset(
            inp_path=data_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=False,
            transforms=None,
            normalise=None
        )
        img_size = 256
    elif dataset == "melanoma_segmentation":
        inp_path = "../datasets/melanoma_segmentation/test/ISBI2016_ISIC_Part1_Test_Data"
        gt_path = "../datasets/melanoma_segmentation/test/ISBI2016_ISIC_Part1_Test_GroundTruth"
        perm_idx = torch.randperm(379)
        val_set_idx = perm_idx[:75]
        train_set_idx = perm_idx[75:]
        valset = MelanomaDataset(
            inp_path=inp_path,
            gt_path=gt_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=False,
            transforms=None,
            normalise=None,
            test=True
        )
        img_size = 512
    else:
        raise ValueError("Invalid value for dataset.")

    return valset, img_size


def compute_metrics(
        net_name: str,
        dataset: str,
        valset: torchvision.datasets,
        img_size: float,
        identifier: str,
        path: str,
        solver: str,
        pretrained: bool = False
) -> tuple[float, float, float, float, float]:
    torch.manual_seed(0)
    checkpoint = os.path.join(path, dataset, "checkpoints", identifier + ".pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if net_name == "conMultUKAN":
        net = conMultUKAN(num_classes=output_dim, input_channels=3, deep_supervision=False, img_size=img_size,
                          patch_size=16,
                          in_chans=3,
                          embed_dims=[128, 160, 256], output_dim=[32, 64, 128], no_kan=False,
                          drop_rate=0., time_dependent=False, non_linearity="softplus",
                          tol=1e-3, adjoint=False, method="rk4", drop_path_rate=0., norm_layer=nn.LayerNorm)
    else:
        raise ValueError(
            "Please pick one of the network architectures"
        )

    model = load_checkpoint(
        checkpoint=checkpoint,
        model=net,
        device=device,
    )

    testloader = torch.utils.data.DataLoader(
        valset, batch_size=1, shuffle=False, num_workers=4
    )
    model.to(device)
    dice = 0.0
    hd95 = 0.0
    acc = 0.0
    counter = 0
    iou = 0.0
    f1 = 0.0
    dice_op = torchmetrics.Dice(average="macro", num_classes=2).to(device)
    f1_op_binary = torchmetrics.F1Score(task="binary").to(device)
    f1_op_multiclass = torchmetrics.F1Score(task="multiclass", num_classes=2).to(device)

    for img, lbl in testloader:
        counter += 1
        img = img.to(device)
        lbl = lbl.to(device)
        out = model(img)
        lbl_detach = torch.argmax(lbl.detach(), dim=1).int() if output_dim == 2 else lbl.detach().int()
        if output_dim == 1:
            out_thres = (torch.sigmoid(out) > 0.5).int()
            f1_op = f1_op_binary
            hd95 += hd95_score(out_thres, lbl_detach)
        elif output_dim == 2:
            out_thres = torch.softmax(out, dim=1)
            f1_op = f1_op_multiclass
            hd95 += hd95_score(out_thres, lbl_detach, 3)
        else:
            raise ValueError(f"Unexpected output_dim={output_dim}, only 1 or 2 is supported.")

        dice += dice_op(out_thres, lbl_detach)
        acc += pixelwise_acc(pred=out_thres, lbl=lbl_detach)
        iou += iou_score(out_thres, lbl_detach)
        f1 += f1_op(out_thres, lbl_detach)
    dice /= counter
    hd95 /= counter
    acc /= counter
    iou /= counter
    f1 /= counter

    total_params = sum(p.numel() for p in net.parameters())
    print(f'Total number of parameters: {total_params}')

    print(
        f"Solver: {solver} - Dice: {dice:.4f} - hd95: {hd95:.2f} - Accuracy: {acc:.4f} - IoU: {iou:.4f} - F1: {f1:.4f}")
    input_size_tuple = (3, img_size, img_size)
    flops, _ = get_model_complexity_info(net, input_size_tuple,
                                         as_strings=False,
                                         print_per_layer_stat=False)
    print(f'Total number of parameters: {total_params}, flops: {flops}')

    return dice, hd95, acc, iou, f1


def load_checkpoint(
        checkpoint: str,
        model: nn.Module,
        device: torch.device,
):
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()

    return model


def write_to_log_file(solver, net_name, log_file, metric_name, metric_value):
    with open(log_file, "a+") as fw:
        if solver is not None:
            line = f"\n{solver} - {metric_name}: {metric_value:.4f}"
        else:
            line = f"\n{net_name} - {metric_name}: {metric_value:.4f}"
        fw.write(line)


if __name__ == "__main__":
    args = parser.parse_args()
    valset, img_size = load_dataset(dataset=args.dataset)
    for identifier, net_name, solver in zip(args.identifier, args.net, args.solver):
        print(identifier, net_name)
        log_file = os.path.join(
            args.path, args.dataset, "log_files", f"{identifier}_METRICS_SOLVERS.txt"
        )
        dice, hd95, acc, iou, f1 = compute_metrics(
            net_name=net_name,
            dataset=args.dataset,
            valset=valset,
            img_size=img_size,
            identifier=identifier,
            path=args.path,
            solver=solver,
            pretrained=args.pretrained
        )
        write_to_log_file(solver=solver, net_name=net_name, log_file=log_file, metric_name="Dice",
                          metric_value=dice)
        write_to_log_file(solver=solver, net_name=net_name, log_file=log_file, metric_name="HD95",
                          metric_value=hd95)
        write_to_log_file(solver=solver, net_name=net_name, log_file=log_file, metric_name="Accuracy",
                          metric_value=acc)
        write_to_log_file(solver=solver, net_name=net_name, log_file=log_file, metric_name="Accuracy",
                          metric_value=iou)
        write_to_log_file(solver=solver, net_name=net_name, log_file=log_file, metric_name="Accuracy",
                          metric_value=f1)
