# Official Implementation of "Implicit U-KAN2.0: Dynamic, Efficient and Interpretable Medical Image Segmentation"[MICCAI 2025]

This is the official implementation of "[Implicit U-KAN2.0: Dynamic, Efficient and Interpretable Medical Image Segmentation](https://papers.miccai.org/miccai-2025/paper/2894_paper.pdf)", accepted by MICCAI 2025. 

![arch](/Users/yiningzhao/Desktop/implicit-conukan/assets/arch.png)

## Installation

Follow these steps to set up the environment:

1. Clone the repository:

   ```bash
   git clone https://github.com/robotMonkeyButler/implicit-conukan.git
   cd conMultUKAN
   ```

2. Create and activate Conda environment

   ```bash
   conda create -n conmultukan python=3.10
   conda activate conmultukan
   pip install -r requirement.txt
   ```

3. Verify the environment by running

   ```bash
   python -c "import torch; print(torch.__verison__)"
   ```



## Datasets

Download the datasets and put them under the repo folder.

## Usage

### Training

To train a model, use the following command

```bash
python train.py --dataset <DATASET> --net <NETWORK> --batch-size <BATCH_SIZE> --epochs <EPOCHS> --lr <LEARNING_RATE>
```

##### Description:

`--epochs`: Number of training epochs (default: 500).

`--batch-size`: Batch size for training (default: 8).

`--dataset`: Dataset to use ( `polyp`, `nuclei`,  `breast_cancer`,  `melanoma_segmentation`).

`--lr`: Learning rate (default: 1e-4).

`--gradient-accumulation`: Mini-batch size for gradient accumulation (default: 1).

`--pretrained`: Boolean to load a pretrained model (default: False).

`--checkpoint`: Path to a checkpoint file (if `--pretrained` is True). 

`--solver`: Solver of ODE block. 

### Testing

To test a model, use the following command

```bash
python test_2d.py --dataset <DATASET> --net <NETWORK> --path ./outputs --identifier <MODEL_IDENTIFIER>
```

##### Description:

`--dataset`: Dataset to use ( `polyp`, `nuclei`,  `breast_cancer`,  `melanoma_segmentation`).

`--path`: output path

`--pretrained`: Boolean to load a pretrained model (default: False).

`--solver`: Solver of ODE block.

`--identifier`: Identifier of training model

e.g.

```bash
python test_2d --dataset polyp --net conMultUKAN --path ./outputs --identifier conMultUKAN-rk4-2024-12-09T041730Z_best
```



#### Output Directory

The results of training and validation are saved in the ./outputs/ directory uner the respective dataset name. The structure is as follows:

```plaintext
./outputs/
   <dataset_name>/
      checkpoints/
         {identifier}.pt
      log_files/
```



#### **Citation**

```
@inproceedings{cheng2025implicit,
  title={Implicit U-KAN2. 0: Dynamic, efficient and interpretable medical image segmentation},
  author={Cheng, Chun-Wun and Zhao, Yining and Cheng, Yanqi and Montoya-Zegarra, Javier A and Sch{\"o}nlieb, Carola-Bibiane and Aviles-Rivero, Angelica I},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={304--314},
  year={2025},
  organization={Springer}
}
```



