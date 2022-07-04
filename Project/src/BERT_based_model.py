# network
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class Network(nn.Module):
    # ProtBert pre-trained model 
    # Rostlab/prot_bert : https://www.biorxiv.org/content/10.1101/2020.07.12.199554v3
    # DistillProtBert : https://www.biorxiv.org/content/10.1101/2022.05.09.491157v1
    def __init__(self, hidden_dims : int, num_classes : int, fixed_layer : int = 10):
        super(Network, self).__init__()
        # self.bert = BertModel.from_pretrained("Rostlab/prot_bert")
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