import os
import random

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class TorinoAquaDataset(Dataset):
    def __init__(self, rootdir='torinoaqua') -> None:
        super().__init__()
        self.rootdir = rootdir
        self.listdir = os.listdir(rootdir)
        self.transfroms = ToTensor()
        
    def __len__(self):
        return len(self.listdir)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.rootdir,self.listdir[index]))
        img = img.convert('RGBA')
        x, y = random.randint(0, img.size[0]-512), random.randint(0, img.size[1]-512)
        label = img.crop((x,y,x+512,y+512))

        egdeLength = 400
        x, y = random.randint(0, 512-egdeLength), random.randint(0, 512-egdeLength)

        temp = label.crop((x, y, x+egdeLength, y+egdeLength)).resize((egdeLength//10,egdeLength//10), Image.Resampling.NEAREST).resize((egdeLength,egdeLength), Image.Resampling.NEAREST)

        
        mask = Image.new('L', temp.size, 0)
        draw = ImageDraw.Draw(mask)
        for i in range(3):
            ew, eh = random.randint(egdeLength//5, egdeLength//3), random.randint(egdeLength//5, egdeLength//3)
            ex, ey = random.randint(0, temp.size[0]-ew), random.randint(0, temp.size[1]-eh)
            draw.ellipse((ex, ey, ex+ew, ey+eh), fill=255)
        temp.putalpha(mask)

        input = label.copy()
        input.paste(temp, (x, y, x+egdeLength, y+egdeLength), temp)
        maskbig = Image.new('L', label.size, 0)
        maskbig.paste(mask, (x, y, x+egdeLength, y+egdeLength), mask)

        input = input.convert('RGB')
        label = label.convert('RGB')
        maskbig = maskbig.convert('L')
        input = self.transfroms(input)
        label = self.transfroms(label)
        maskbig = self.transfroms(maskbig)
        return {'input':input, 'label':label, 'mask':maskbig}

if __name__ == '__main__':
    dataset = TorinoAquaDataset()
    fig, axes = plt.subplots(3, 3, figsize=(50,20))
    for i in range(3):
        data = dataset[i]
        axes[i, 0].set_title('input')
        axes[i, 0].imshow(data['input'].numpy().transpose(1,2,0))
        axes[i, 1].set_title('label')
        axes[i, 1].imshow(data['label'].numpy().transpose(1,2,0))
        axes[i, 2].set_title('mask')
        axes[i, 2].imshow(data['mask'].numpy().transpose(1,2,0))
    plt.show()
    print(data['input'].shape, data['label'].shape, data['mask'].shape)

