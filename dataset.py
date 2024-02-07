import os
import random

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class TorinoAquaDataset(Dataset):
    def __init__(self, rootdir='ImageDatasets', no_mask=0.5, num_sample=-1, factor=(10,50)) -> None:
        '''
            rootdir: relative path of the dataset
            no_mask: 0.1 for 10% no masked images, -1 for 100% masked images
            num_sample: -1 for full
            factor: resize factor
        '''
        super().__init__()
        self.listdir = []
        for dirpath, dirs, filenames in os.walk(rootdir):
            for f in filenames:
                self.listdir.append(os.path.join(dirpath, f))
        self.transfroms = ToTensor()
        self.no_mask = no_mask
        self.len = len(self.listdir) if num_sample==-1 else min(len(self.listdir), num_sample)
        self.factor = factor
        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img = Image.open(self.listdir[index])
        assert img.size[0] >= 512 and img.size[1] >= 512, f'Check image size ({self.listdir[index]}) in dataset.'
        img = img.convert('RGBA')
        x, y = random.randint(0, img.size[0]-512), random.randint(0, img.size[1]-512)
        label = img.crop((x,y,x+512,y+512))
        input = label.copy()
        maskbig = Image.new('L', label.size, 0)
        if random.random()<self.no_mask:
            input = input.convert('RGB')
            label = label.convert('RGB')
            maskbig = maskbig.convert('L')
            input = self.transfroms(input)
            label = self.transfroms(label)
            maskbig = self.transfroms(maskbig)
            return {'input':input, 'label':label, 'mask':maskbig}

        egdeLength = 400
        x, y = random.randint(0, 512-egdeLength), random.randint(0, 512-egdeLength)
        factor = random.randint(*self.factor)
        temp = label.crop((x, y, x+egdeLength, y+egdeLength)).resize((egdeLength//factor,egdeLength//factor), Image.Resampling.NEAREST).resize((egdeLength,egdeLength), Image.Resampling.NEAREST)

        mask = Image.new('L', temp.size, 0)
        draw = ImageDraw.Draw(mask)
        for i in range(3):
            ew, eh = random.randint(50, egdeLength-50), random.randint(50, egdeLength-50)
            ex, ey = random.randint(0, temp.size[0]-ew), random.randint(0, temp.size[1]-eh)
            draw.ellipse((ex, ey, ex+ew, ey+eh), fill=255)
        temp.putalpha(mask)

        input.paste(temp, (x, y, x+egdeLength, y+egdeLength), temp)
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
    fig, axes = plt.subplots(5, 3, figsize=(50,20))
    for i in range(5):
        data = dataset[i]
        axes[i, 0].set_title('input')
        axes[i, 0].imshow(data['input'].numpy().transpose(1,2,0))
        axes[i, 1].set_title('label')
        axes[i, 1].imshow(data['label'].numpy().transpose(1,2,0))
        axes[i, 2].set_title('mask')
        axes[i, 2].imshow(data['mask'].numpy().transpose(1,2,0))
    plt.show()
    print(len(dataset))
    print(data['input'].shape, data['label'].shape, data['mask'].shape)

