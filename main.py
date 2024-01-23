from dataset import TorinoAquaDataset
from model import AutoEncoder
import os
import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoEncoder().to(device)
lossfn = torch.nn.BCEWithLogitsLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-4)

dataset = TorinoAquaDataset()
dataloader = DataLoader(dataset, 1, True, num_workers=os.cpu_count())
epochs = 50
for epoch in range(epochs):
    print('Epoch:', epoch+1)
    for i, data in enumerate(tqdm(dataloader)):
        inputs = data['input'].to(device)
        label = data['mask'].to(device)
        logits = model(inputs)
        loss = lossfn(logits, label)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if (i+1)%50==0:
            print(loss)
            fig, axes = plt.subplots(1, 3, figsize=(50,20))
            axes[0].set_title('input')
            axes[0].imshow(inputs[0].cpu().numpy().transpose(1,2,0))
            axes[1].set_title('label')
            axes[1].imshow(label[0].cpu().numpy().transpose(1,2,0))
            axes[2].set_title('predict')
            axes[2].imshow(F.sigmoid(logits)[0].cpu().detach().numpy().transpose(1,2,0))
            plt.savefig(f'Train/Epoch{epoch}_{i}')