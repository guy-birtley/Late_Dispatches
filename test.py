import torch
from torch.utils.data import DataLoader
import numpy as np
from model_classes import FoundationDataset, Model2
from tqdm import tqdm

if __name__ == "__main__": # for multiple spawns

    test_data_raw = np.load(r'cache\test.npz')
    test_dataset = FoundationDataset(**test_data_raw)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True, pin_memory=True)


    device = torch.device("xpu")
    model = Model2()


    # data = np.load(r"C:\Python Projects\ML Transactions\data_cache\obs_2025.npz")
    # test_dataset = FoundationDataset(data["temporal_val"], data["dense_val"], data["mask_val"], data["Y_val"])

    checkpoint = torch.load(r"cache\model_checkpoint.pt")

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    model.eval() # Set to evaluation mode
    test_loss = 0
    criterion = torch.nn.BCEWithLogitsLoss() # loss fn

    with torch.no_grad(): # Disable gradient calculation
        for batch_count, test_batch in enumerate(tqdm(test_dataloader, desc="Training")):
            X = test_batch['X'].to(device)
            mask = test_batch['mask'].to(device)
            Y = test_batch['Y'].to(device)
            
            out = model(X, mask)
            loss = criterion(out.logits, Y)
            test_loss += loss.item()
            print(f"\n Batch {batch_count} loss: {loss.item():.3f}")
            
    print(f"--- Random Sample Validation Loss: {(loss / len(test_dataloader)):.3f} ---")

    model.train() # return to training mode
