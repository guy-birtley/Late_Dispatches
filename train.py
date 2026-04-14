import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from helper import tprint
from model_classes import FoundationDataset, Model2, ModelDense


if __name__ == "__main__": # for multiple spawns

    tprint('Loading data')


    train_dataset = FoundationDataset(*train_data_list) #change test to train if run preprocessing again
    val_dataset = FoundationDataset(*val_data_list)

    tprint('Getting moment foundation model')
    
    #initialise model
    #model = Model2()
    model = ModelDense()

    criterion = torch.nn.BCEWithLogitsLoss() # loss fn
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    #lower triangle causal mask - used if training multiple classifiers on same history
    # causal_mask = torch.tril(torch.ones((context_window, context_window), dtype=torch.bool)).unsqueeze(0).expand(batch_size, -1, -1)

    # training data shape: [batches, preditors, context length (512)]
    # Y shape : [batches, ]

    tprint('Putting model onto graphics card')
    device = torch.device("xpu")
    model = model.to(device)

    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0) 
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)


    tprint('Training')
    for batch_count, batch in enumerate(tqdm(train_dataloader, desc="Training")):
        #use gpu for faster computation
        X= batch['X'].to(device, non_blocking = True)
        mask= batch['mask'].to(device, non_blocking = True)
        dense = batch['dense'].to(device, non_blocking = True)
        Y  = batch['Y'].to(device, non_blocking = True)

        # forward [batch_size, n_channels, context length]
        #output = model(X, mask, dense) # get logits from custom forward pass
        output = model(dense)
        # backward
        loss = model.criterion(output, Y) # compare logits to Y
        optimizer.zero_grad() # reset gradients
        loss.backward() # calculate gradients with backwards pass
        optimizer.step() # update weights
        
        print(f"\n Batch {batch_count} loss: {loss.item():.3f}")

        # if batch_count % 5 == 0 and batch_count != 0: # Validate every 5 batches
        #     model.eval() # Set to evaluation mode
        #     val_loss = 0
            
        #     with torch.no_grad(): # Disable gradient calculation
        #         for val_batch in val_dataloader:
        #             vX = val_batch['X'].to(device)
        #             vmask = val_batch['mask'].to(device)
        #             vY = val_batch['Y'].to(device)
                    
        #             v_out = model(vX, vmask)
        #             v_loss = criterion(v_out.logits, vY)
        #             val_loss += v_loss.item()
                    
        #     avg_val_loss = val_loss / len(val_dataloader)
        #     tprint(f"--- Random Sample Validation Loss: {avg_val_loss:.3f} ---")
            
        #     model.train() # return to training mode

        if batch_count % 100 == 0:
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