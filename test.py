import torch
from torch.utils.data import DataLoader
import numpy as np
from train_transformer import FoundationDataset, TransformerModel
from tqdm import tqdm
from helper import tprint, y_labels

if __name__ == "__main__": # for multiple spawns

    tprint('loading training data')
    test_data_raw = np.load(r'cache\test.npz')
    test_dataset = FoundationDataset(**test_data_raw)

    tprint('loading model')
    device = torch.device("xpu")
    model = TransformerModel(test_data_raw['dense'].shape[-1], test_data_raw['X'].shape[-1], len(y_labels))
    checkpoint = torch.load(r"cache\model_checkpoint.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval() # Set to evaluation mode

    batch_size = 16
    
    tprint('iterating through test set')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loss = 0.0
    y_hat = []
    with torch.no_grad(): # Disable gradient calculation
        for test_batch in tqdm(test_dataloader, desc="Predicting Test Set"):
            X = test_batch['X'].to(device)
            mask = test_batch['mask'].to(device)
            dense_in = test_batch['dense'].to(device)
            Y = test_batch['Y'].to(device)
            
            output = model(X, mask, dense_in)
            loss = model.criterion(output, Y)
            if loss.item() > 100:
                print("--- Exploding Batch Detected ---")
                print('Mean X across batches', torch.mean(X, dim = 0))
                print('Mean dense across batches', torch.mean(dense_in, dim = 0))
                print('Active tokens', {mask.sum(dim=-1)})
                print('Y_true', Y)
                print('Y_hat ', torch.argmax(output, dim=1).cpu())
            
            y_hat.append(torch.argmax(output, dim=1).cpu())
            test_loss += loss.item()

            
    print(f'Test loss:', (test_loss / len(test_dataloader)))
    print(f'Test accuracy:', (torch.cat(y_hat, dim=0).numpy() == test_data_raw['Y']).mean())

    print(torch.cat(y_hat, dim=0))
    print(test_data_raw['Y'])

    model.train() # return to training mode
