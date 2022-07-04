import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score

def evaluate(
    valid_loader : torch.utils.data.DataLoader, 
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    class_label = [0,1,2,3,4,5,6,7,8,9]
    ):

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

            output = model(input_ids, input_ids.size(0))
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
    print(classification_report(total_label, total_pred, labels = [i for i in range(len(class_label))]))
    print("\n# total test f1 score : {:.2f} and test loss : {:.3f}".format(valid_f1, valid_loss))

    fig , ax = plt.subplots()
    fig.set_size_inches(13, 8)

    sns.heatmap(
            conf_mat / np.sum(conf_mat, axis = 1),
            annot = True,
            fmt = '.2f',
            cmap = 'Blues',
            xticklabels=len(class_label),
            yticklabels=len(class_label)
    )
    plt.show()
    
    return valid_loss, valid_acc, f1_score(total_label, total_pred, average = "macro")
