
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
from momentfm import MOMENTPipeline


class FoundationDataset(Dataset):
    def __init__(self, X, dense, mask, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.dense = torch.tensor(dense, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return {
            "X": self.X[idx],
            "dense": self.dense[idx],
            "mask": self.mask[idx],
            "Y": self.Y[idx]
        }
    

class embModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.loc_dims = 6
        self.portal_dims = 11
        self.predictors = 19

        #get pretrained model from MOMENT
        self.foundation_model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-Base", #"AutonLab/MOMENT-1-large", 
        model_kwargs={
            'task_name': 'classification',
            'n_channels': self.predictors, #if want to embed with more than 1 channel this must increase here
            'num_class': 1 #binary classifier for now
        },)

        self.foundation_model.init()

        #freeze all foundation parameters but the head
        for name, param in self.foundation_model.named_parameters():
            param.requires_grad = (name.split('.')[0] == 'head')
        
        #columns to embed from left of X input
        self.emb_layers = nn.ModuleList([
            nn.Embedding(self.loc_dims, 1),
            nn.Embedding(self.portal_dims, 1)
        ])
        
        self.criterion = torch.nn.BCEWithLogitsLoss() # loss fn


    #set forward pass to embed categorical columns first
    def forward(self, x, mask):
        # x: [B, T, C]
        embs = []
        
        # embed first channels
        for i, layer in enumerate(self.emb_layers):
            emb = layer(x[:, :, i].round().long())   # [B, T, emb_dim]
            embs.append(emb)

        # concatenate embeddings and remaining channels
        x = torch.cat(embs + [x[:, :, len(self.emb_layers):]], dim=2)
        
        # swap final columns for foundation model
        x = x.permute(0, 2, 1)

        # pass through foundation model
        out = self.foundation_model(x_enc=x, attention_mask=mask)
        
        return out
    
def tprint(text):
    print(pd.Timestamp.now(), text)