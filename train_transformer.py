import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from helper import tprint, y_labels
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
        #set hyper parameters
        self.found_neurons = 128 #output neurons of foundation model
        self.dense_neurons = 128 #output neurons of dense layer
        self.dense_dropout = 0.3 #dropout rate of dense layer
        self.min_sequence_length = 16 #minimum sequence length to run through transformer layer (otherwise 0)

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
            param.requires_grad = True #"encoder.final_layer_norm" in name # or "encoder.block.11" in name # cannot unfreeze last block :(

        # process output of MOMENT
        self.found_branch = nn.Sequential(
            nn.Linear(768, self.found_neurons), # project 768 output MOMENT to moment_neurons
            nn.LeakyReLU(0.01) # nn.ReLU()
        )

        self.dense_branch = nn.Sequential(
            nn.Linear(dense_predictors, self.dense_neurons), # project 13 dense layer parameters to self.dense_neurons
            nn.LeakyReLU(0.01), # nn.ReLU(),
            nn.Dropout(self.dense_dropout) # dropout to reduce reliance on dense inputs
        )
        
        self.combine_branch = nn.Sequential(
            nn.Linear(self.found_neurons + self.dense_neurons + 1, 128), # dense layer after concatenation
            nn.LeakyReLU(0.01), #nn.ReLU(),
            nn.Linear(128, 16), # another dense layer
            nn.LeakyReLU(0.01), #nn.ReLU(),
            nn.Linear(16, Y_classes) # output logit score for each Y class
        )

        # self.test_temporal_direct = nn.Sequential(
        #     nn.Linear(self.dense_neurons, 64),
        #     nn.LeakyReLU(0.01),
        #     nn.Linear(64, 16),
        #     nn.LeakyReLU(0.01), #nn.ReLU(),
        #     nn.Linear(16, Y_classes)
        # )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, X, mask, dense_in):
    #     dense_branch_out = self.dense_branch(dense_in)
    #     return self.test_temporal_direct(dense_branch_out)

    # def full_forward():
        #foundation stream
        x = X.permute(0, 2, 1) # swap dimensions for moment input so now [batch, channels, context window]
        # only pass input sequences with more than 5 transactions - short sequences produce nans and should be predicted by dense layer instead
        is_long_enough = (mask.sum(dim=-1) >= self.min_sequence_length) #mask batch where sequences less than min_sequence length
        found_branch_out = torch.zeros((x.size(0), self.found_neurons), device=x.device) #initialise output of zeros
        if is_long_enough.any():
            #get embeddings where sequence is long enough [batch, channels, moment output dimension]
            found_raw = self.foundation_model(x_enc=x[is_long_enough], input_mask=mask[is_long_enough]).embeddings
            found_branch_out[is_long_enough] = self.found_branch(found_raw)
        used_moment = is_long_enough.float().unsqueeze(-1) #binary indicator if that batch element used transformer or not

        # dense stream
        dense_branch_out = self.dense_branch(dense_in)

        # concatenate streams
        combined = torch.cat([found_branch_out, dense_branch_out, used_moment], dim=-1)

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
    model = TransformerModel(train_data_raw['dense'].shape[-1], train_data_raw['X'].shape[-1], len(y_labels))

    #lower triangle causal mask - used if training multiple classifiers on same history
    # causal_mask = torch.tril(torch.ones((context_window, context_window), dtype=torch.bool)).unsqueeze(0).expand(batch_size, -1, -1)

    tprint('Putting model onto graphics card')
    device = torch.device("xpu")
    model = model.to(device)

    batch_size = 16

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0) 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    tprint('Training')
    for batch_count, batch in enumerate(tqdm(train_dataloader, desc="Training")):
    # batch = next(iter(train_dataloader))
    # for batch_count in range(500):
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
                        print(f"{name} gradient: {param.grad.abs().mean().item()}")
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