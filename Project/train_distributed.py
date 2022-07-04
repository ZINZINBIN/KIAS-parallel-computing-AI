from typing import Optional
import torch
import torch.nn
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import re, os
import random
from tqdm.auto import tqdm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score


# dataset
class CustomDataset(Dataset):
    def __init__(self, dataset, df:pd.DataFrame, task_type : str = "train"):
        self.dataset = dataset
        self.df = df
        self.task_type = task_type
        self.labels = []

        if self.task_type == "train":
            for cls_idx in df['cls_idx'].values:
                self.labels.append(cls_idx)
            self.labels = np.array(self.labels).reshape(-1,)
        
    def __getitem__(self, idx):
       
        item = {
                key : val[idx].clone().detach() for key, val in self.dataset.items()
        }

        if self.task_type == "train":
            item['label'] = torch.tensor(self.labels[idx])
            return item
        else:
            return item

    def __len__(self):
        return len(self.df)
    
class CustomDatasetUpdated(Dataset):
    def __init__(self, df, tokenizer, max_len, max_pred, mask_prob,):
        super().__init__()
        self.df = df # data
        self.len = len(self.df)
        self.max_pred = max_pred # max tokens of prediction
        self.mask_prob = mask_prob # masking probability
        self.tokenizer = tokenizer # tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        item = self.process(item)
        return item
           
    def process(self, row):
        seq = row['sequence'] # 'ACGEF...'
        class_num = row['cls_idx']
        
        # random crop
        if len(seq) >= self.max_len+5:
            idx = random.randint(0,len(seq)-self.max_len-5)
            seq = seq[idx:idx+self.max_len]
        seq = re.sub(r"[UZOB]", "X", seq)
        tokens = list(seq)

        # Tokenize
        ret = self.tokenizer(" ".join(tokens),
                            return_tensors = 'pt',
                            max_length = self.max_len,
                            padding = 'max_length',
                            truncation=True,
                            add_special_tokens=True)
        ret = {k: v.squeeze() for k,v in ret.items()}
        ret['label'] = torch.tensor(class_num)
        return ret
    
    def __len__(self):
        return len(self.df)

# model
class Network(nn.Module):
    def __init__(self, hidden_dims : int, num_classes : int, fixed_layer : int = 10):
        super(Network, self).__init__()
        self.bert = BertModel.from_pretrained("yarongef/DistilProtBert")
        self.classifier = nn.Sequential(
            nn.Linear(1024,hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims,hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, num_classes)
        )
        
        self.num_classes = num_classes
        self.fixed_layer = fixed_layer
        
        for name, p in self.bert.base_model.named_parameters():
            if name.startswith('pooler'):
                pass
            elif len(name.split('.')) > 3 and name.split('.')[2].isdigit() and int(name.split('.')[2]) > fixed_layer:
                pass
            else:
                p.requires_grad = False

    def forward(self, input_ids : torch.Tensor, token_ids : torch.Tensor, att_mask : torch.Tensor)->torch.Tensor:
        bert_output = self.bert(input_ids, token_ids, att_mask)
        o = bert_output.pooler_output
        x = self.classifier(o)
        return x

def set_random_seeds(random_seed : int = 42):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def get_distributed_loader(train_dataset : Dataset, valid_dataset : Dataset, num_replicas : int, rank : int, num_workers : int, batch_size : int = 32):
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=num_replicas, rank = rank, shuffle = True)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=num_replicas, rank = rank, shuffle = True)
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, sampler = train_sampler, num_workers = num_workers, pin_memory = False)
    valid_loader = DataLoader(valid_dataset, batch_size = batch_size, sampler = valid_sampler, num_workers = num_workers, pin_memory = False)

    return train_loader, valid_loader


def train_epoch_per_procs(
    rank : int, 
    world_size : int, 
    batch_size : Optional[int],
    model : torch.nn.Module,
    train_dataset : Dataset,
    valid_dataset : Dataset,
    random_seed : int = 42,
    resume : bool = True,
    loss_fn = None,
    model_filepath : str = "./weights/distributed.pt"
    ):

    device = torch.device("cuda:{}".format(rank))
    set_random_seeds(random_seed)

    model.to(device)
    ddp_model = DDP(model, device_ids = [rank], output_device=rank)

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss(reduction = "mean")
        
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr = 1e-3)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.95)

    if not os.path.isfile(model_filepath) and dist.get_rank() == 0:
        torch.save(model.state_dict(), model_filepath)

    dist.barrier()

    if resume == True:
        map_location = {"cuda:0":"cuda:{}".format(rank)}
        ddp_model.load_state_dict(torch.load(model_filepath, map_location=map_location), strict = False)

    train_loader, valid_loader = get_distributed_loader(train_dataset, valid_dataset, num_replicas=world_size, rank = rank, num_workers = 16, batch_size = batch_size)

    # train process
    model.train()
    train_loss = 0
    train_acc = 0
    
    total_pred = np.array([])
    total_label = np.array([])
    
    for batch_idx, items in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids, token_ids, att_mask, target = items['input_ids'], items['token_type_ids'], items['attention_mask'], items['label']
        
        input_ids = input_ids.to(device)
        token_ids = token_ids.to(device)
        att_mask = att_mask.to(device)
        target = target.to(device)

        output = model(input_ids, token_ids, att_mask)
        loss = loss_fn(output, target)

        loss.backward()

        # gradient cliping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        train_loss += loss.item()

        pred = torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1]
        train_acc += pred.eq(target.view_as(pred)).sum().item() / input_ids.size(0) 

        total_pred = np.concatenate((total_pred, pred.cpu().numpy().reshape(-1,)))
        total_label = np.concatenate((total_label, target.cpu().numpy().reshape(-1,)))
       
    if scheduler:
        scheduler.step()

    train_loss /= (batch_idx + 1)
    train_acc /= (batch_idx + 1)

    train_f1 = f1_score(total_label, total_pred, average = "macro")
    
    dist.barrier()

    # valid process
    model.eval()
    valid_loss = 0
    valid_acc = 0
    
    total_pred = np.array([])
    total_label = np.array([])
    
    for batch_idx, items in enumerate(valid_loader):
        with torch.no_grad():
            optimizer.zero_grad()
            input_ids, token_ids, att_mask, target = items['input_ids'], items['token_type_ids'], items['attention_mask'], items['label']

            input_ids = input_ids.to(device)
            token_ids = token_ids.to(device)
            att_mask = att_mask.to(device)
            target = target.to(device)

            output = model(input_ids, token_ids, att_mask)
            loss = loss_fn(output, target)

            valid_loss += loss.item()

            pred = torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1]
            valid_acc += pred.eq(target.view_as(pred)).sum().item() / input_ids.size(0) 

            total_pred = np.concatenate((total_pred, pred.cpu().numpy().reshape(-1,)))
            total_label = np.concatenate((total_label, target.cpu().numpy().reshape(-1,)))

    valid_loss /= (batch_idx + 1)
    valid_acc /= (batch_idx + 1)
    valid_f1 = f1_score(total_label, total_pred, average = "macro")
    
    
    dist.barrier()

    return train_loss, train_acc, train_f1, valid_loss, valid_acc, valid_f1

import time

def train_per_proc(
    rank : int, 
    world_size : int, 
    batch_size : Optional[int],
    model : torch.nn.Module,
    train_dataset : Dataset,
    valid_dataset : Dataset,
    resume : bool = True,
    loss_fn = None,
    model_filepath : str = "./weights/distributed.pt",
    num_epoch : int = 64,
    verbose : Optional[int] = 8,
    save_best_only : bool = False,
    save_best_dir : str = "./weights/best.pt"
):
    dist.init_process_group("nccl", rank = rank, world_size = world_size)

    train_loss_list = []
    valid_loss_list = []
    
    train_acc_list = []
    valid_acc_list = []
    
    train_f1_list = []
    valid_f1_list = []

    best_acc = 0
    best_f1 = 0
    best_epoch = 0
    best_loss = np.inf
    
    # radomized seed
    random_seed = 42 + dist.get_rank()
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    start_time = time.time()

    for epoch in tqdm(range(num_epoch)):

        train_loss, train_acc, train_f1, valid_loss, valid_acc, valid_f1 = train_epoch_per_procs(
            rank,
            world_size,
            batch_size,
            model,
            train_dataset,
            valid_dataset,
            random_seed = random_seed,
            resume = True,
            loss_fn = loss_fn,
            model_filepath = model_filepath
        )


        if dist.get_rank() == 0:
            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)

            train_acc_list.append(train_acc)
            valid_acc_list.append(valid_acc)
            
            train_f1_list.append(train_f1)
            valid_f1_list.append(valid_f1)
            
            if verbose:
                if epoch % verbose == 0:
                    
                    end_time = time.time()
                    dt = end_time - start_time
                    
                    print("epoch : {}, train loss : {:.3f}, valid loss : {:.3f}, train f1 : {:.3f}, valid f1 : {:.3f}, dt : {:.2f}".format(
                        epoch+1, train_loss, valid_loss, train_f1, valid_f1, dt
                    ))

            if save_best_only:
                if best_loss > valid_loss:
                    best_acc = valid_acc
                    best_f1 = valid_f1
                    best_loss = valid_loss
                    best_epoch  = epoch
                    torch.save(model.state_dict(), save_best_dir)

            torch.save(model.state_dict(), model_filepath)
    
    if dist.get_rank() == 0:
        # print("\n============ Report ==============\n")
        print("training process finished, best loss : {:.3f} and best f1 : {:.3f}, best epoch : {}".format(
            best_loss, best_f1, best_epoch
        ))
        
    dist.destroy_process_group()

    return  train_loss_list, train_acc_list, valid_loss_list, valid_acc_list


def train_distributed(
    world_size : int, 
    batch_size : Optional[int],
    model : torch.nn.Module,
    train_dataset : Dataset,
    valid_dataset : Dataset,
    random_seed : int = 42,
    resume : bool = True,
    loss_fn = None,
    model_filepath : str = "./weights/distributed.pt",
    num_epoch : int = 64,
    verbose : Optional[int] = 8,
    save_best_only : bool = False,
    save_best_dir : str = "./weights/distributed_best.pt"
    ):

    if world_size > torch.cuda.device_count():
        world_size = torch.cuda.device_count()

    mp.spawn(
        train_per_proc,
        args = (world_size,batch_size, model, train_dataset,valid_dataset, resume, loss_fn, model_filepath, num_epoch, verbose, save_best_only, save_best_dir),
        nprocs = world_size,
        join = True
    )

    
def evaluate(
    valid_loader : torch.utils.data.DataLoader, 
    model : torch.nn.Module,
    optimizer : Optional[torch.optim.Optimizer],
    loss_fn : torch.nn.Module,
    device : str = "cpu"
    ):

    model.eval()
    model.to(device)
    valid_loss = 0
    valid_acc = 0

    total_pred = np.array([])
    total_label = np.array([])
    
    for batch_idx, items in enumerate(test_loader):
        with torch.no_grad():
            input_ids, token_ids, att_mask, target = items['input_ids'], items['token_type_ids'], items['attention_mask'], items['label']
            input_ids = input_ids.to(device)
            token_ids = token_ids.to(device)
            att_mask = att_mask.to(device)
            target = target.to(device)

            output = model(input_ids, token_ids, att_mask)
            loss = loss_fn(output, target)

            valid_loss += loss.item()

            pred = torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1]
            valid_acc += pred.eq(target.view_as(pred)).sum().item() / input_ids.size(0) 

            total_pred = np.concatenate((total_pred, pred.cpu().numpy().reshape(-1,)))
            total_label = np.concatenate((total_label, target.cpu().numpy().reshape(-1,)))

    valid_loss /= (batch_idx + 1)
    valid_acc /= (batch_idx + 1)
    valid_f1 = f1_score(total_label, total_pred, average = "macro")

    conf_mat = confusion_matrix(total_label, total_pred)
    
    print("############### Classification Report ####################")
    print(classification_report(total_label, total_pred, labels = [i for i in range(model.num_classes)]))
    print("\n# total test f1 score : {:.2f} and test loss : {:.3f}".format(valid_f1, valid_loss))

    return valid_loss, valid_acc, f1_score(total_label, total_pred, average = "macro")


def compute_focal_loss(inputs:torch.Tensor, gamma:float):
    p = torch.exp(-inputs)
    loss = (1-p) ** gamma * inputs
    return loss.mean()

# focal loss object
class FocalLossLDAM(nn.Module):
    def __init__(self, weight : Optional[torch.Tensor] = None, gamma : float = 0.1):
        super(FocalLossLDAM, self).__init__()
        assert gamma >= 0, "gamma should be positive"
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs : torch.Tensor, target : torch.Tensor)->torch.Tensor:
        return compute_focal_loss(F.cross_entropy(inputs, target, reduction = 'mean', weight = self.weight), self.gamma)

if __name__ == "__main__":
    
    # parameter setting
    use_focal_loss = False
    padding = True
    truncation = True
    add_special_tokens = True
    batch_size = 64
    hidden_dims = 128
    n_criteria = 5000
    save_dir = "./weight/best.pt"
    model_filepath = "./weight/last.pt"
    
    max_len=400
    max_pred=5
    mask_prob=0.15
    
    fixed_layer = 10
    
    num_epoch = 32
    verbose = 1
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    
    # torch device state
    print("torch device avaliable : ", torch.cuda.is_available())
    print("torch current device : ", torch.cuda.current_device())
    print("torch device num : ", torch.cuda.device_count())
    
    # torch cuda initialize and clear cache
    torch.cuda.init()
    torch.cuda.empty_cache()

    
    # dataset path 설정
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
    
    from transformers import BertModel, BertTokenizer
    import re

    from sklearn.model_selection import train_test_split
    
    # data['sequence'] = data['sequence'].apply(lambda x : " ".join(x))
    df_train, df_test = train_test_split(data, test_size = 0.2, random_state = 42)
    df_train, df_valid = train_test_split(df_train, test_size = 0.3, random_state = 42)

    tokenizer = BertTokenizer.from_pretrained("yarongef/DistilProtBert", do_lower_case=False)
    
    print("train dataset : ", len(df_train))
    print("valid dataset : ", len(df_valid))
    print("test dataset : ", len(df_test))

    
#     train_tokenize = tokenizer(
#         list(df_train['sequence']), 
#         return_tensors = 'pt',
#         max_length = max_len,
#         padding = padding,
#         truncation = truncation,
#         add_special_tokens = add_special_tokens
#     )

#     valid_tokenize = tokenizer(
#         list(df_valid['sequence']), 
#         return_tensors = 'pt',
#         max_length = max_len,
#         padding = padding,
#         truncation = truncation,
#         add_special_tokens = add_special_tokens
#     )

#     test_tokenize = tokenizer(
#         list(df_test['sequence']), 
#         return_tensors = 'pt',
#         max_length = max_len,
#         padding = padding,
#         truncation = truncation,
#         add_special_tokens = add_special_tokens
#     )

#     train_data = CustomDataset(train_tokenize, df_train, "train")
#     valid_data = CustomDataset(valid_tokenize, df_valid, "train")
#     test_data = CustomDataset(test_tokenize, df_test, "train")
        

    # dataset and dataloader 
    train_data = CustomDatasetUpdated(df_train,tokenizer,max_len=max_len,max_pred=max_pred,mask_prob=mask_prob)
    valid_data = CustomDatasetUpdated(df_valid,tokenizer,max_len=max_len,max_pred=max_pred,mask_prob=mask_prob)
    test_data = CustomDatasetUpdated(df_test,tokenizer,max_len=max_len,max_pred=max_pred,mask_prob=mask_prob)
    
    model = Network(hidden_dims = hidden_dims, num_classes = len(cls_names), fixed_layer = fixed_layer)
    
    if use_focal_loss:
        loss_fn = FocalLossLDAM(weight = None, gamma=0.5)
    else: 
        loss_fn = torch.nn.CrossEntropyLoss(reduction = "mean")


    # 분산 학습 진행
    train_distributed(
        world_size = torch.cuda.device_count(),
        batch_size = batch_size,
        model = model,
        train_dataset=train_data,
        valid_dataset=valid_data,
        resume = True,
        loss_fn = loss_fn,
        model_filepath = model_filepath,
        num_epoch=num_epoch,
        verbose = verbose,
        save_best_only=True,
        save_best_dir=save_dir
    )
    
    
   # load best weight
    model.load_state_dict(torch.load(save_dir), strict = False)
    
    device = torch.device("cuda:0")
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = True, num_workers = 8, pin_memory = True)

    test_loss, test_acc, conf_mat = evaluate(
        test_loader,
        model,
        None,
        loss_fn, 
        device,
    )