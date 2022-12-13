
import vonenet
import torchvision
import torch
import deeplake
import numpy as np
import matplotlib.pyplot as plt
import skimage
import skimage.io
import torch.nn as nn
import tqdm



def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = [correct[:k].sum().item() for k in topk]
        return res

model = vonenet.get_model(model_arch='simple', pretrained=False, noise_mode=None).module

checkpoint = torch.load('/home/jasmeet/vonenet/epoch_02_v.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
print(f'model {checkpoint} loaded')
# img_path='/home/jasmeet/Downloads/tiny-imagenet-200-2/val/images/val_1008.JPEG'
# img = skimage.io.imread(img_path)/255.0
data_path = '/home/jasmeet/vonenet/val_gaussian'
normalize = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                             std=[0.5, 0.5, 0.5])
dataset = torchvision.datasets.ImageFolder(data_path,
    torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        normalize,
    ]))

data_loader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True, num_workers=20, pin_memory=True)
record = 0
loss = nn.CrossEntropyLoss(size_average=False)
l = 0
accuracy1 = 0
accuracy5 = 0
model.eval()
with torch.no_grad():
    for (inp, target) in tqdm.tqdm(data_loader, desc="Validation"):
       result=model(inp)
       l +=  loss(result, target).item()
       p1,p5 = accuracy(result,target,topk=(1,5))
       accuracy1 += p1
       accuracy5 += p5

print("Accuracy 1 :",accuracy1)
print("Accuracy 5 :",accuracy5) 
print("Loss :",l/10000)      