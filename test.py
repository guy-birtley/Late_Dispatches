import torch
from torch.utils.data import DataLoader
import numpy as np
from helper import FoundationDataset, embModel
from tqdm import tqdm

device = torch.device("xpu")

data = np.load(r"C:\Python Projects\ML Transactions\data_cache\obs_2025.npz")
val_dataset = FoundationDataset(data["temporal_val"], data["dense_val"], data["mask_val"], data["Y_val"])

checkpoint = torch.load("moment_checkpoint_latest.pt")
model = embModel()
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=True, pin_memory=True)

model.eval() # Set to evaluation mode
val_loss = 0
criterion = torch.nn.BCEWithLogitsLoss() # loss fn

with torch.no_grad(): # Disable gradient calculation
    for batch_count, val_batch in enumerate(tqdm(val_dataloader, desc="Training")):
        vX = val_batch['X'].to(device)
        vmask = val_batch['mask'].to(device)
        vY = val_batch['Y'].to(device)
        
        v_out = model(vX, vmask)
        v_loss = criterion(v_out.logits, vY)
        val_loss += v_loss.item()
        print(f"\n Batch {batch_count} loss: {v_loss.item():.3f}")
        
avg_val_loss = val_loss / len(val_dataloader)
print(f"--- Random Sample Validation Loss: {avg_val_loss:.3f} ---")

model.train() # return to training mode