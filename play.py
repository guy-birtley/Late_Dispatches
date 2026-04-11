import torch

device = torch.device("xpu")


model = torch.load("moment_checkpoint_latest.pt", map_location=device)