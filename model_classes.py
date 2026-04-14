
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

        # Freeze foundation parameters
        for param in self.foundation_model.parameters():
            param.requires_grad = False
        
        # Define the actual layers to be trained
        # MOMENT-Base typically has an embedding dimension of 512
        # We have 5 channels, so flattened/pooled it is often 512
        self.moment_proj = nn.Linear(768, 64)
        self.dense_proj = nn.Linear(13, 64) # Assuming your 13 metadata features
        
        self.fc1 = nn.Linear(64 + 64, 64)
        self.fc2 = nn.Linear(64, 16)
        self.out_layer = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, X, mask, dense_in):
        # 1. Prepare X for MOMENT: (Batch, Channels, Length)
        x = X.permute(0, 2, 1)

        # 2. Get embeddings: returns object with .embeddings attribute
        # Shape: [batch, channels, d_model] or [batch, d_model] depending on pooling
        moment_out = self.foundation_model(x_enc=x, input_mask=mask)
        
        # Most MOMENT tasks pool to [batch, d_model]. 
        # If it returns [batch, channels, d_model], we mean-pool the channels:
        z_moment = moment_out.embeddings
        if z_moment.dim() == 3:
            z_moment = z_moment.mean(dim=1)

        # 3. Process both streams
        z_moment = self.relu(self.moment_proj(z_moment))
        z_dense = self.relu(self.dense_proj(dense_in))

        # 4. Concatenate and Head
        combined = torch.cat([z_moment, z_dense], dim=-1)
        
        x = self.relu(self.fc1(combined))
        x = self.relu(self.fc2(x))
        return self.out_layer(x)
    
    def predict(self, x, mask, dense_in):
        logits = self.forward(x, mask, dense_in)
        return torch.sigmoid(logits) # 0-1 probability

class ModelDense(nn.Module):
    #model just testing a dense network without temporal features
    def __init__(self):
        super().__init__()
        self.dense_input = nn.Linear(13, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 16)
        self.out_layer = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, dense_in):
        x = self.relu(self.dense_input(dense_in))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out_layer(x)
    
    def predict(self, dense_in):
        logits = self.forward(dense_in)
        return torch.sigmoid(logits) # 0-1 probability
