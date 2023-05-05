# Multimodal Pre-training for Sequential Recommendation via Contrastive Learning


#### This code repository is for paper \#3086.



## Installation
We first need to create a python=3.7 virtualenv and activate it.\
Then, we install some dependencies.
```bash
pip install -r requirements.txt
```


## Dataset Preparation
Due to the space limit on the google drive folders, we can't attach all datasets in this package. \
Hence, we provide the Pantry dataset, which can be downloaded from [Link](https://drive.google.com/drive/folders/1dUoj5nPqhLXR1DoLHBgwWvkcsbjnIXdh?usp=share_link). 
We will provide the other two datasets once the paper is published. 
After downloading the dataset, please put them under './dataset/pantry/'.

For example, the complete dataset folder structure of Pantry is as follows:
```
dataset/pantry
    |── pantry.inter                     # user-item interaction data
    ├── desc_emb.pkl                     # item text feature
    └── image_emb_15.pkl                 # item image feature
```
"15" refers to the number of image word tokens. By default, the number of image word tokens is 15.

## Training MP4SR
Our proposed method is a two stage framework. Hence, we need to first run *pre-training* and then run *fine-tuning* on the same dataset.

### Pre-training
Pre-train MP4SR from scratch:
```bash
CUDA_VISIBLE_DEVICES=1 python run_recbole.py --gpu_id=1 --dataset=pantry --model=MP4SR --train_stage=pretrain --num_imgtokens=15 --learning_rate=0.001 --train_batch_size=1024 --lambda=0.01 --proj=False
```

The output is a pre-trained checkpoint named as {MP4SR-Date-Time} saved in the folder './saved/'. \
--lambda is used to balance Modality-specific Next Item Prediction loss (NIP) and Cross Modality Contrastive Loss
(CMCL). 


### Fine-tuning
1. Find the pre-trained checkpoint file name from the .log file or './saved/' folder. For example, the checkpoint file is named as 'MP4SR-Aug-18-2022_19-38-01.pth'.
Then, fine-tune the model with the pre-trained model for one set of hyper-parameters:
```bash
python run_recbole.py --dataset=pantry --model=MP4SR --train_stage='finetune' --num_imgtokens=15 --pretrained_path='./saved/MP4SR-Aug-18-2022_19-38-01.pth' --learning_rate=0.0001 --train_batch_size=1024 --weight_decay=0.001
```
The training log file is written under './log/'. \
--pretrained_path defines the saved pre-trained checkpoint.

2. Or, we could fine-tune with pre-trained model for full sets of hyper-parameters:
```bash
python run_hyper.py --dataset=pantry --model=MP4SR --train_stage='finetune' --num_imgtokens=15 --pretrained_path='./saved/MP4SR-Aug-18-2022_19-38-01.pth' --output_file='./log_tune/pantry.result' --params_file=hyper.test
```
--output_file specifies the output directory of the hyper-parameter tuning results. \
--params_file defines the search space for hyper-parameter tuning.
