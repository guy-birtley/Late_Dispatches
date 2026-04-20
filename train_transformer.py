import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from helper import tprint
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from momentfm import MOMENTPipeline



class FoundationDataset(Dataset):
    def __init__(self, **kwargs):
        self.X = torch.tensor(kwargs["X"], dtype=torch.float32)
        self.dense = torch.tensor(kwargs["dense"], dtype=torch.float32)
        self.mask = torch.tensor(kwargs["mask"], dtype=torch.float32)
        self.Y = torch.tensor(kwargs["Y"], dtype=torch.long)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return {
            "X": self.X[idx],
            "dense": self.dense[idx],
            "mask": self.mask[idx],
            "Y": self.Y[idx]
        }


class TransformerModel(nn.Module):
    def __init__(self, dense_predictors, X_predictors, Y_classes):
        super().__init__()

        self.foundation_model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-Base",
            model_kwargs={
                'task_name': 'embedding',
                'n_channels': X_predictors
            },
        )
        self.foundation_model.init()

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
            nn.Linear(dense_predictors, dense_neurons), # project 13 dense layer parameters to dense_neurons
            nn.BatchNorm1d(dense_neurons), # normalise batches to prevent dense layer dominating
            nn.ReLU(),
            nn.Dropout(0.5) # dropout to reduce reliance on dense inputs
        )
        
        self.combine_branch = nn.Sequential(
            nn.Linear(found_neurons + dense_neurons, 64), # dense layer after concatenation
            nn.ReLU(),
            nn.Linear(64, 16), # another dense layer
            nn.ReLU(),
            nn.Linear(16, Y_classes) # output logit score for each Y class
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


if __name__ == "__main__": # for multiple spawns

    tprint('Loading data')
    train_data_raw = np.load(r'cache\train.npz')

    train_dataset = FoundationDataset(**train_data_raw) #change test to train if run preprocessing again

    tprint('Getting moment foundation model')
    
    #initialise model
    model = TransformerModel(train_data_raw['dense'].shape[-1], train_data_raw['X'].shape[-1], train_data_raw['Y'].shape[-1])

    #lower triangle causal mask - used if training multiple classifiers on same history
    # causal_mask = torch.tril(torch.ones((context_window, context_window), dtype=torch.bool)).unsqueeze(0).expand(batch_size, -1, -1)

    # training data shape: [batches, preditors, context length (512)]
    # Y shape : [batches, ]

    tprint('Putting model onto graphics card')
    device = torch.device("xpu")
    model = model.to(device)

    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0) 
    

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    tprint('Training')
    for batch_count, batch in enumerate(tqdm(train_dataloader, desc="Training")):
        #use gpu for faster computation
        X = batch['X'].to(device, non_blocking = True)
        mask = batch['mask'].to(device, non_blocking = True)
        dense = batch['dense'].to(device, non_blocking = True)
        Y  = batch['Y'].to(device, non_blocking = True)

        # forward [batch_size, n_channels, context length]
        output = model(X, mask, dense) # output logit score of each Y class

        # backward
        loss = model.criterion(output, Y) # compare logits to Y (should be high number for true class and low for everything else)
        optimizer.zero_grad() # reset gradients
        loss.backward() # calculate gradients with backwards pass
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #clip gradients to prevent explosion
        
        # print gradients
        if batch_count % 10 == 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        print(f"{name} grad: {param.grad.abs().mean().item():.5f}")
                    else:
                        print(f"{name} has NO GRADIENT!")

        optimizer.step() # update weights
        accuracy = (torch.argmax(output, dim=1) == Y).float().mean()
        print(f"\n Batch {batch_count}; loss {loss.item():.3f}; accuracy {accuracy}")
        


        if batch_count % 1000 == 0:
            tprint("Saving checkpoint...")
            # Create a dictionary to store everything needed to resume
            checkpoint = {
                'batch_count': batch_count,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }
            # Overwrite the same file to save space
            torch.save(checkpoint, r"cache\model_checkpoint.pt")
            tprint("Checkpoint saved.")

tprint("Saving final copy...")
# Create a dictionary to store everything needed to resume
checkpoint = {
    'batch_count': batch_count,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
}
torch.save(checkpoint, r"cache\model_checkpoint.pt")