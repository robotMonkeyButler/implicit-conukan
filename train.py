import os
import torch
import torch.nn as nn
import torchmetrics
import torchvision
import datetime
import argparse
from tqdm import tqdm
import albumentations as A
from utils.data_loader import (
    PolypDataset,
    NucleiDataset,
    BreastCancerDataset,
    MelanomaDataset,
)

import ml_collections
from ptflops import get_model_complexity_info
import time

from implicit_conukan.archs_conmultukan import conMultUKAN
from utils.transforms import RandomRotation, RandomVerticalFlip, RandomHorizontalFlip
from utils.train_utils import save_loss_to_file
from utils.metrics import iou_score


class EarlyStopping:

    def __init__(self, patience=20, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = float('inf')

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.best_loss = val_loss


parser = argparse.ArgumentParser(description="implicit conmultukan experiments")
parser.add_argument(
    "--epochs",
    default=500,
    type=int,
    metavar="N",
    help="number of total epochs to run",
)
parser.add_argument("--batch-size", default=8, type=int, help="batch size")
parser.add_argument(
    "--dataset",
    type=str,
    choices=["polyp", "nuclei", "breast_cancer", "melanoma_segmentation"],
    help="Dataset to use. Choose from: polyp, nuclei, breast_cancer, melanoma_segmentation.",
)
parser.add_argument(
    "--block",
    type=str,
    default="PLN",
    help="Block types: PLN, RSE, DSE, INC, PSP",
)
parser.add_argument(
    "--gradient-accumulation",
    default=1,
    type=int,
    help="mini-batch size for gradient accumulation",
)
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument(
    "--sode-decoder", default=False, type=bool, help="SODE decoder for TransUNet"
)
parser.add_argument("--seed", default=0, type=int, help="Random seed")
parser.add_argument(
    "--pretrained", default=False, type=bool, help="Load pretrained model"
)
parser.add_argument("--checkpoint", type=str, help="Checkpoint ID for pretrained model")
parser.add_argument("--solver", type=str, default="rk4", help="Solver")


def load_checkpoint(
        checkpoint: str,
        model: nn.Module,
        device: torch.device,
):
    model = model
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()

    return model


def main():
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.dataset == "polyp":
        data_path = (
            "../datasets/Kvasir-SEG"
        )
        inp_path = os.path.join(data_path, "images/train")
        gt_path = os.path.join(data_path, "masks/train")
        perm_idx = torch.randperm(880)
        val_set_idx = perm_idx[:176]
        train_set_idx = perm_idx[176:]
        transforms = torchvision.transforms.Compose(
            [RandomVerticalFlip(p=0.5), RandomHorizontalFlip(p=0.5)],
        )

        mean = [0.55894187, 0.32169286, 0.23561553]
        std = [0.30564184, 0.21442238, 0.17755656]
        normalise = torchvision.transforms.Normalize(mean=mean, std=std)

        trainset = PolypDataset(
            inp_path=inp_path,
            gt_path=gt_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=True,
            transforms=transforms,
            normalise=None
        )
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
    elif args.dataset == "nuclei":
        data_path = (
            "../datasets/nuclei/train"
        )
        perm_idx = torch.randperm(536)
        val_set_idx = perm_idx[:107]
        train_set_idx = perm_idx[107:]
        mean = [0.1890, 0.1551, 0.1707]
        std = [0.2960, 0.2433, 0.2637]
        transforms = torchvision.transforms.Compose(
            [RandomVerticalFlip(p=0.5), RandomHorizontalFlip(p=0.5)],
        )
        normalise = torchvision.transforms.Compose(
            [torchvision.transforms.Normalize(mean=mean, std=std)]
        )
        trainset = NucleiDataset(
            inp_path=data_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=True,
            transforms=transforms,
            normalise=None
        )
        valset = NucleiDataset(
            inp_path=data_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=False,
            transforms=None,
            normalise=None
        )
        img_size = 256

    elif args.dataset == "breast_cancer":
        data_path = "../datasets/breast_cancer"
        perm_idx = torch.randperm(452)
        val_set_idx = perm_idx[:90]
        train_set_idx = perm_idx[90:]
        transforms = torchvision.transforms.Compose(
            [RandomVerticalFlip(p=0.5), RandomHorizontalFlip(p=0.5)],
        )
        trainset = BreastCancerDataset(
            inp_path=data_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=True,
            transforms=transforms,
            normalise=None
        )
        valset = BreastCancerDataset(
            inp_path=data_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=False,
            transforms=None,
            normalise=None
        )
        img_size = 256
    elif args.dataset == "melanoma_segmentation":
        inp_path = "../datasets/melanoma_segmentation/train/ISBI2016_ISIC_Part1_Training_Data"
        gt_path = "../datasets/melanoma_segmentation/train/ISBI2016_ISIC_Part1_Training_GroundTruth"
        perm_idx = torch.randperm(900)
        val_set_idx = perm_idx[:180]
        train_set_idx = perm_idx[180:]
        transforms = torchvision.transforms.Compose(
            [RandomVerticalFlip(p=0.5), RandomHorizontalFlip(p=0.5)],
        )
        trainset = MelanomaDataset(
            inp_path=inp_path,
            gt_path=gt_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=True,
            transforms=transforms,
            normalise=None
        )
        valset = MelanomaDataset(
            inp_path=inp_path,
            gt_path=gt_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=False,
            transforms=None,
            normalise=None
        )
        img_size = 512
    else:
        raise ValueError("Invalid value for dataset.")

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = conMultUKAN(num_classes=output_dim, input_channels=3, deep_supervision=False, img_size=img_size,
                          patch_size=16,
                          in_chans=3,
                          no_kan=False, embed_dims=[128, 256, 512], output_dim=[32, 64, 128],
                          drop_rate=0., time_dependent=False, non_linearity="softplus",
                          tol=1e-3, adjoint=False, method="rk4", drop_path_rate=0., norm_layer=nn.LayerNorm)
                          
    if args.pretrained:
        net = load_checkpoint(checkpoint=args.checkpoint, model=net, device=device)
        print("Using pretrained model from {}".format(args.checkpoint))
    else:
        def weights_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        net.apply(weights_init)

    net.to(device)

    total_params = sum(p.numel() for p in net.parameters())
    print(f'Total number of parameters: {total_params}')
    date_str = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")

    logfile = os.path.join(
        "./outputs",
        args.dataset,
        "log_files",
        "implicit_conukan" + "-" + (args.solver if args.solver else "") + "-" + date_str + ".txt",
    )
    os.makedirs(os.path.dirname(logfile), exist_ok=True)

    with open(logfile, "a+") as fw:
        fw.write(f'Total number of parameters: {total_params}\n')

    input_size_tuple = (3, img_size, img_size)
    macs, params = get_model_complexity_info(net, input_size_tuple, as_strings=False,
                                             print_per_layer_stat=False, verbose=False)
    paramsm= params / 1e6
    print(f'Total number of parameters: {paramsm:.2f}M')
    gflops = macs / 1e9
    print(f'Total number of GFLOPs: {gflops:.2f}')
    
    if macs is None:
        macs = 0
    flops = macs * 2

    print(f'Total FLOPs: {flops}')
    with open(logfile, "a+") as fw:
        fw.write(f'Total FLOPs: {flops}\n')

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                                                                patience=5,
                                                                                                verbose=True)

    date_str = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")

    filename = os.path.join(
        "./outputs",
        args.dataset,
        "checkpoints",
        "implicit_conukan" + "-" + args.solver + "-" + date_str + ".pt",
    )
    filename_best = os.path.join(
        "./outputs",
        args.dataset,
        "checkpoints",
        "implicit_conukan" + "-" + args.solver + "-" + date_str + "_best.pt",
    )

    logfile = os.path.join(
        "./outputs",
        args.dataset,
        "log_files",
        "implicit_conukan" + "-" + args.solver + "-" + date_str + ".txt",
    )

    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    early_stopping = EarlyStopping(patience=20, verbose=True, path=filename_best)

    try:
        run(
            args=args,
            logfile=logfile,
            optimizer=optimizer,
            trainloader=trainloader,
            valloader=valloader,
            net=net,
            scheduler=scheduler,
            filename=filename,
            filename_best=filename_best,
            device=device,
            flops_per_input=flops,
            img_size=img_size,
            early_stopping=early_stopping,
        )
    except Exception as e:
        print(f"Training interrupted: {e}")
        with open(logfile, "a+") as fw:
            fw.write(f"Training interrupted: {e}\n")
        torch.save(net.state_dict(), filename + "_interrupt.pt")


def run(
        args,
        logfile,
        optimizer,
        trainloader,
        valloader,
        net,
        scheduler,
        filename,
        filename_best,
        device,
        flops_per_input,
        img_size,
        early_stopping,
):
    losses = []
    val_losses = []
    best_val_loss = float('inf')

    line = f"Learning rate: {args.lr} - Batch size: {args.batch_size} - Accumulate Gradient: {args.gradient_accumulation} - Epochs: {args.epochs} - Pretrained: {args.pretrained}"
    with open(logfile, "a+") as fw:
        fw.write(line + "\n")
    print(line)

    for epoch in range(args.epochs):
        accumulated = 0
        running_loss = 0.0
        optimizer.zero_grad()
        net.train()
        total_time = 0.0
        total_flops = 0.0

        for data in tqdm(trainloader):
            start_time = time.time()

            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)

            criterion = torch.nn.BCEWithLogitsLoss()
            loss = criterion(outputs, labels) / args.gradient_accumulation

            loss.backward()
            accumulated += 1
            if accumulated == args.gradient_accumulation:
                optimizer.step()
                optimizer.zero_grad()
                accumulated = 0

            running_loss += loss.item()
            batch_time = time.time() - start_time
            total_time += batch_time
            batch_flops = flops_per_input * inputs.size(0)
            total_flops += batch_flops

        if accumulated > 0:
            optimizer.step()
            optimizer.zero_grad()
            accumulated = 0

        batched_loss = running_loss / (len(trainloader) * args.gradient_accumulation)
        print("\n Training loss: " + str(batched_loss))
        save_loss_to_file(logfile=logfile, eval_type="Training", loss=batched_loss)
        losses.append(batched_loss)

        if total_time > 0:
            average_gflops = total_flops / total_time / 1e9
            print(f'Epoch [{epoch + 1}/{args.epochs}] Average GFLOPS: {average_gflops:.2f}')
            with open(logfile, "a+") as fw:
                fw.write(f'Epoch [{epoch + 1}/{args.epochs}] Average GFLOPS: {average_gflops:.2f}\n')
        else:
            print('Total time is zero, cannot compute GFLOPS.')

        with torch.no_grad():
            net.eval()
            running_loss = 0.0

            for data in valloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)

                criterion = torch.nn.BCEWithLogitsLoss()
                loss = criterion(outputs, labels) / args.gradient_accumulation

                running_loss += loss.item()

            batched_val_loss = running_loss / len(valloader)
            print(" Validation loss: " + str(batched_val_loss))
            save_loss_to_file(
                logfile=logfile, eval_type="Validation", loss=batched_val_loss
            )
            val_losses.append(batched_val_loss)

            torch.save(net.state_dict(), filename)
            if batched_val_loss < best_val_loss:
                best_val_loss = batched_val_loss
                torch.save(net.state_dict(), filename_best)
                print(f"Best model saved with val_loss score: {best_val_loss:.4f}")

        early_stopping(batched_val_loss, net)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            with open(logfile, "a+") as fw:
                fw.write("Early stopping triggered. Stopping training.\n")
            break

        if scheduler is not None:
            scheduler.step(batched_val_loss)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main()
