
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from momentfm import MOMENTPipeline

class FoundationDataset(Dataset):
    def __init__(self, **kwargs):
        self.X = torch.tensor(kwargs["X"], dtype=torch.float32)
        self.dense = torch.tensor(kwargs["dense"], dtype=torch.float32)
        self.mask = torch.tensor(kwargs["mask"], dtype=torch.float32)
        self.Y = torch.tensor(kwargs["Y"], dtype=torch.float32)

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

class Model2(nn.Module):
    def __init__(self):
        super().__init__()

        self.foundation_model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-Base",
            model_kwargs={
                'task_name': 'embedding',
                'n_channels': 5
            },
        )
        self.foundation_model.init()
        
        # # Freeze foundation parameters
        # for param in self.foundation_model.parameters():
        #     param.requires_grad = False

        #train only final block
        for name, param in self.foundation_model.named_parameters():
            param.requires_grad = "encoder.final_layer_norm" in name # or "encoder.block.11" in name # cannot unfreeze last block :(

        found_neurons = 256
        dense_neurons = 64

        # process output of MOMENT
        self.found_branch = nn.Sequential(
            nn.Linear(768, found_neurons), # project 768 output MOMENT to moment_neurons
            nn.ReLU(),
            nn.BatchNorm1d(found_neurons), #normalise batch
        )

        self.dense_branch = nn.Sequential(
            nn.Linear(13, dense_neurons), # project 13 dense layer parameters to dense_neurons
            nn.BatchNorm1d(dense_neurons), # normalise batches to prevent dense layer dominating
            nn.ReLU(),
            nn.Dropout(0.5) # dropout to reduce reliance on dense inputs
        )
        
        self.combine_branch = nn.Sequential(
            nn.Linear(found_neurons + dense_neurons, 64), # dense layer after concatenation
            nn.ReLU(),
            nn.Linear(64, 16), # another dense layer
            nn.ReLU(),
            nn.Linear(16, 4) # output logit score for each Y class
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, X, mask, dense_in):

        #foundation stream
        x = X.permute(0, 2, 1) # Change dimensions of X for MOMENT: [Batch, Channels, Length]
        found_out_raw = self.foundation_model(x_enc=x, input_mask=mask).embeddings #get embeddings [batch, channels, moment output dimension]
        found_branch_out = self.found_branch(found_out_raw)
        
        # dense stream
        dense_branch_out = self.dense_branch(dense_in)

        # concatenate streams
        combined = torch.cat([found_branch_out, dense_branch_out], dim=-1)
        #combined = torch.cat([found_branch_out, torch.zeros_like(dense_branch_out)], dim=-1) # test to see what it learns if no outputs

        return self.combine_branch(combined)
    
    def predict(self, x, mask, dense_in):
        logits = self.forward(x, mask, dense_in)
        return torch.softmax(logits, dim=1) # predict class membership (to predict probability of each class membership, use sigmoid)
