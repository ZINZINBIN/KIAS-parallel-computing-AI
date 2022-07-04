import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from typing import Optional, Union
from tqdm.auto import tqdm

def train_per_epoch(
    train_loader : torch.utils.data.DataLoader, 
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    max_norm_grad : float = 1.0,
    model_type : str = "self-attention" # self-attention / ProtBert
    ):

    if model_type != "self-attention":
        model_type = "ProtBert"

    model.train()
    model.to(device)

    train_loss = 0
    train_acc = 0

    total_pred = np.array([])
    total_label = np.array([])

    for batch_idx, items in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids, token_ids, att_mask, target = items['input_ids'], items['token_type_ids'], items['attention_mask'], items['label']
        
        input_ids = input_ids.to(device)
        target = target.to(device)

        if model_type == "self-attention":
            output = model(input_ids, input_ids.size(0))
        else:
            token_ids = token_ids.to(device)
            att_mask = att_mask.to(device)
            output = model(input_ids, token_ids, att_mask)

        loss = loss_fn(output, target)

        loss.backward()

        # gradient cliping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm_grad)

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

    return train_loss, train_acc, train_f1

def valid_per_epoch(
    valid_loader : torch.utils.data.DataLoader, 
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    model_type : str = "self-attention"
    ):

    if model_type != "self-attention":
        model_type = "ProtBert"

    model.eval()
    model.to(device)
    valid_loss = 0
    valid_acc = 0

    total_pred = np.array([])
    total_label = np.array([])

    for batch_idx, items in enumerate(valid_loader):
        with torch.no_grad():
            optimizer.zero_grad()
            input_ids, token_ids, att_mask, target = items['input_ids'], items['token_type_ids'], items['attention_mask'], items['label']
        
            input_ids = input_ids.to(device)
            target = target.to(device)

            if model_type == "self-attention":
                output = model(input_ids, input_ids.size(0))
            else:
                token_ids = token_ids.to(device)
                att_mask = att_mask.to(device)
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
 
    return valid_loss, valid_acc, valid_f1

def train(
    train_loader : torch.utils.data.DataLoader, 
    valid_loader : torch.utils.data.DataLoader,
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn :Optional[torch.nn.Module]= None,
    device : str = "cpu",
    num_epoch : int = 64,
    verbose : Optional[int] = 8,
    save_best_only : bool = False,
    save_best_dir : str = "./weights",
    criteria : str = "f1_score",
    max_grad_norm : float = 1.0,
    model_type : str = "self-attention"
):

    train_loss_list = []
    valid_loss_list = []
    
    train_acc_list = []
    valid_acc_list = []

    train_f1_list = []
    valid_f1_list = []

    best_acc = 0
    best_f1 = 0
    best_epoch = 0
    best_loss = torch.inf

    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss(reduction = 'mean')

    for epoch in tqdm(range(num_epoch), desc = "training process"):

        train_loss, train_acc, train_f1 = train_per_epoch(
            train_loader, 
            model,
            optimizer,
            scheduler,
            loss_fn,
            device,
            max_grad_norm,
            model_type
        )

        valid_loss, valid_acc, valid_f1 = valid_per_epoch(
            valid_loader, 
            model,
            optimizer,
            loss_fn,
            device,
            model_type
        )

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)

        train_f1_list.append(train_f1)
        valid_f1_list.append(valid_f1)

        if verbose:
            if epoch % verbose == 0:
                print("epoch : {}, train loss : {:.3f}, valid loss : {:.3f}, train acc : {:.3f}, valid acc : {:.3f}, train f1 : {:.3f}, valid f1 : {:.3f}".format(
                    epoch+1, train_loss, valid_loss, train_acc, valid_acc, train_f1, valid_f1
                ))

        if save_best_only:
            if criteria == "f1_score" and best_f1 < valid_f1:
                best_f1 = valid_f1
                best_loss = valid_loss
                best_epoch  = epoch

                if not os.path.exists(save_best_dir):
                    os.mkdir(save_best_dir)

                torch.save(model.state_dict(), save_best_dir + "/best.pt")

            if criteria == "loss" and best_loss > valid_loss:
                best_f1 = valid_f1
                best_loss = valid_loss
                best_epoch  = epoch

                if not os.path.exists(save_best_dir):
                    os.mkdir(save_best_dir)

                torch.save(model.state_dict(), save_best_dir + "/best.pt")

    # print("\n============ Report ==============\n")
    print("training process finished, best loss : {:.3f} and best f1 : {:.3f}, best epoch : {}".format(
        best_loss, best_f1, best_epoch
    ))

    return  train_loss_list, train_acc_list, train_f1_list,  valid_loss_list,  valid_acc_list, valid_f1_list