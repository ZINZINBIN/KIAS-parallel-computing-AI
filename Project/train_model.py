import os
import torch
import torch.nn as nn
import numpy as np
import argparse 
import matplotlib as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from src.loss import FocalLossLDAM
from src.self_attention import SelfAttention
from src.BERT_based_model import Network
from src.train import train
from src.dataset import CustomDataset
from src.utils import plot_learning_curve
from src.evaluate import evaluate

parser = argparse.ArgumentParser(description="training Protein Structure/Functional Classification model")
parser.add_argument("--model_type", str, default = "self-attention") # self-attention / ProtBert
parser.add_argument("--use_focal_loss", bool, default = True)
parser.add_argument("--batch_size", int, default = 64)
parser.add_argument("--n_criteria", int, default = 5000, help = "effective number of data for each class")
parser.add_argument("--num_epoch", int, default = 1)
parser.add_argument("--verbose", int, default = 1)
parser.add_argument("--save_dir", str, default = "./weight/best.pt")
parser.add_argument("--model_filepath", str, default = "./weight/last.pt")
parser.add_argument("--max_len", int, default = 400)
parser.add_argument("--focal_gamma", float, default = 0.5)
parser.add_argument("--lr", float, default = 1e-3)

# torch device state
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()

# device allocation
if(torch.cuda.device_count() >= 1):
    device = "cuda:0" 
else:
    device = 'cpu'

args = vars(parser.parse_args())

# parameter setting
model_type = args["model_type"]
use_focal_loss = args["use_focal_loss"]
padding = True
truncation = True
add_special_tokens = True
batch_size = args["batch_size"]

n_criteria = args["hidden_dims"]
save_dir = args["save_dir"]
model_filepath = args["model_filepath"]
max_len=args["max_len"]
max_pred=5
mask_prob=0.15

num_epoch = args["num_epoch"]
verbose = args["verbose"]
focal_gamma = args["focal_gamma"]
lr = args["lr"]


if __name__ == "__main__":

    # set dataset path
    PATH = "./dataset/"

    df_pdb_char = pd.read_csv(PATH + "pdb_data_no_dups.csv")
    df_pdb_seq = pd.read_csv(PATH + "pdb_data_seq.csv")
    
    # select protein class
    df_pdb_seq = df_pdb_seq[df_pdb_seq["macromoleculeType"]=="Protein"]
    df_pdb_char = df_pdb_char[df_pdb_char["macromoleculeType"] == "Protein"]

    # merge two dataframe
    df_pdb_seq.set_index("structureId")
    df_pdb_char.set_index("structureId")
    df = pd.merge(left = df_pdb_seq, right = df_pdb_char)
    
    cols = [
        "structureId", 
        "sequence", 
        "chainId", 
        "macromoleculeType", 
        "classification", 
    ]
    
    df = df[cols].dropna(axis = 0)
    counts = df.classification.value_counts()
    effective_counts = counts[counts > n_criteria]
    
    # data filtering
    data = df[df.classification.isin(effective_counts.keys())]
    data = data.drop_duplicates(subset=["classification","sequence"])
    
    seq_len_counts = data.sequence.apply(lambda x : len(x)).value_counts()
    
    # class to label
    counts = data.classification.value_counts()
    cls_names = counts.keys().values

    class2label = {}
    label2class = {}

    for idx, name in enumerate(cls_names):
        class2label[name] = idx
        label2class[idx] = name

    def ConvertLabel2Class(x : int):
        cls = label2class[x]
        return cls

    def ConvertClass2Label(x : str):
        label = class2label[x]
        return label

    data['cls_idx'] = data['classification'].apply(lambda x : ConvertClass2Label(x))

    df_train, df_test = train_test_split(data, test_size = 0.2, random_state = 42)
    df_train, df_valid = train_test_split(df_train, test_size = 0.3, random_state = 42)

    tokenizer = BertTokenizer.from_pretrained("yarongef/DistilProtBert", do_lower_case=False)
    
    # data info
    print("train dataset : ", len(df_train))
    print("valid dataset : ", len(df_valid))
    print("test dataset : ", len(df_test))

    n_tokens = len(tokenizer.vocab)

    train_data = CustomDataset(df_train,tokenizer,max_len=max_len,max_pred=max_pred,mask_prob=mask_prob)
    valid_data = CustomDataset(df_valid,tokenizer,max_len=max_len,max_pred=max_pred,mask_prob=mask_prob)
    test_data = CustomDataset(df_test,tokenizer,max_len=max_len,max_pred=max_pred,mask_prob=mask_prob)

    train_loader = DataLoader(train_data, batch_size =batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    valid_loader = DataLoader(valid_data, batch_size =batch_size,  shuffle = True, num_workers = 4, pin_memory = True)
    test_loader = DataLoader(test_data, batch_size =batch_size, shuffle = True, num_workers = 4, pin_memory = True)


    if model_type == "self-attention":
        hidden_dims = 128
        embedd_dims = 128
        model = SelfAttention(batch_size, len(cls_names), hidden_dims, n_tokens, embedd_dims)
    elif model_type == "ProtBert":
        hidden_dims = 128
        fixed_layer = 10
        model = Network(hidden_dims = hidden_dims, num_classes = len(cls_names), fixed_layer = fixed_layer)
    else:
        hidden_dims = 128
        fixed_layer = 10
        model = Network(hidden_dims = hidden_dims, num_classes = len(cls_names), fixed_layer = fixed_layer)
    
    
    if use_focal_loss:
        loss_fn = FocalLossLDAM(weight = None, gamma=0.5)
    else: 
        loss_fn = torch.nn.CrossEntropyLoss(reduction = "mean")

    # optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.95)

    print("\n################# training process #################\n")

    train_loss, train_acc, train_f1, valid_loss, valid_acc, valid_f1 = train(
        train_loader,
        valid_loader,
        model,
        optimizer,
        scheduler,
        loss_fn,
        device,
        num_epoch,
        verbose,
        save_best_only = True,
        save_best_dir = save_dir,
        criteria = "f1_score",
        max_grad_norm = 1.0,
        model_type = model_type
    )

    plot_learning_curve(train_loss, valid_loss, train_f1, valid_f1, save_dir = "./image/learning_curve.png")

    # load best weight
    model.load_state_dict(torch.load(save_dir), strict = False)

    test_loss, test_acc, conf_mat = evaluate(
        test_loader,
        model,
        optimizer,
        loss_fn, 
        device,
    )