import vonenet
import torchvision
import torch
import deeplake
import numpy as np
import matplotlib.pyplot as plt

v1_model = vonenet.get_model(model_arch=None, pretrained=False, noise_mode=None).module

# v1_model = vonenet.get_model(model_arch=None, pretrained=False, noise_mode=None, image_size=32, visual_degrees=3, sf_max=5, stride=1, ksize=15).module
# v1_model = vonenet.get_model(model_arch='resnet50_ns', pretrained=True).module

print(v1_model)

data_path = '/home/jasmeet/vonenet/tiny-imagenet-200/val'

bsize = 16
crop = 256 # 48  256
px = 224 # 32  224

normalize = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                             std=[0.5, 0.5, 0.5])
dataset = torchvision.datasets.ImageFolder(data_path,
    torchvision.transforms.Compose([
        torchvision.transforms.Resize(crop),
        torchvision.transforms.CenterCrop(px),
        torchvision.transforms.ToTensor(),
        normalize,
    ]))

data_loader = torch.utils.data.DataLoader(dataset, batch_size=bsize, shuffle=True, num_workers=6, pin_memory=True)

dataloader_iterator = iter(data_loader)
X, _ = next(dataloader_iterator)
print(X.shape)

activations = v1_model(X)
print(activations.shape)

im_ind = 15

input_im = np.moveaxis(X[im_ind].numpy(), 0, -1)
input_im = input_im / 2 + 0.5

fig, ax = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(7, 7)
im_h = ax.imshow(input_im)
im_h.set_clim([0, 1])
ax.set_axis_off()
plt.show()

np.argsort(np.std(activations[im_ind].numpy().reshape((512, -1)), axis=1))


v1_ind=np.array([427, 208])

fig, ax = plt.subplots(nrows=1, ncols=len(v1_ind))
fig.set_size_inches(15,5)
for v1_i, v1_ind_ in enumerate(v1_ind):
    v1_k = v1_model.simple_conv_q0.weight[v1_ind_,:,:,:].numpy().mean(axis=0)
    v1_k = v1_k / np.amax(np.abs(v1_k))/2+0.5
    im_h=ax[v1_i].imshow(v1_k, cmap='gray')
    ax[v1_i].set_xlim([0, px])
    im_h.set_clim([0, 1])
    ax[v1_i].set_axis_off()
plt.show()


# fig, ax = plt.subplots(nrows=1, ncols=len(v1_ind))
# fig.set_size_inches(15,15)
# max_activations = np.amax(activations[im_ind].numpy())/np.sqrt(2)
# for v1_i, v1_ind_ in enumerate(v1_ind):
#     v1_im = activations[im_ind,v1_ind_].numpy()
#     v1_im = v1_im / max_activations
#     im_h=ax[v1_i].imshow(v1_im, cmap='gray')
#     im_h.set_clim([0, 1])
#     ax[v1_i].set_axis_off()
# plt.show()
#
#
# num_channels=256
# max_columns = 16
#
# fig, ax = plt.subplots(nrows=num_channels//max_columns, ncols=max_columns)
#
# fig.set_size_inches(15,15)
# for i in range(num_channels):
#     v1_k = v1_model.simple_conv_q0.weight[i,:,:,:].numpy().mean(axis=0)
#     v1_k = v1_k / np.amax(np.abs(v1_k))/2+0.5
#     im_h=ax[i//max_columns, np.mod(i,max_columns)].imshow(v1_k, cmap='gray')
# #     ax[i//num_channels, np.mod(i,num_channels)].set_xlim([0, 223])
#     im_h.set_clim([0, 1])
#     ax[i//max_columns, np.mod(i,max_columns)].set_axis_off()
# plt.show()


visual_degrees = 8
image_size = 224

nyquist_f = 1/(visual_degrees/image_size)/2 / np.sqrt(2)

print(nyquist_f)