import skimage
import skimage.io
import matplotlib.pyplot as plt
import os
import shutil
img_path='/home/jasmeet/Downloads/tiny-imagenet-200-2/val/images/val_1008.JPEG'
img = skimage.io.imread(img_path)/255.0

def plotnoise(img, mode, r, c, i):
    #plt.subplot(r,c,i)
    if mode is not None:
        gimg = skimage.util.random_noise(img, mode=mode)
        #plt.imshow(gimg)
        plt.imsave('test.png', gimg)
    else:
        #plt.imshow(img)
        plt.imsave('test.png', img)
    plt.title(mode)
    plt.axis("off")

#plt.figure(figsize=(18,24))
# r=4
# c=2
# plotnoise(img, "gaussian", r,c,1)
# plotnoise(img, "localvar", r,c,2)
# plotnoise(img, "poisson", r,c,3)
# plotnoise(img, "salt", r,c,4)
# plotnoise(img, "pepper", r,c,5)
# plotnoise(img, "s&p", r,c,6)
# plotnoise(img, "speckle", r,c,7)
# plotnoise(img, None, r,c,8)
#plt.show()

source = r'/home/jasmeet/Downloads/tiny-imagenet-200/val/images/'
files = os.listdir(source)
dest = r'/home/jasmeet/vonenet/val_s&p/images/'
for file_name in files:
    img_path=os.path.join(source, file_name)
    img = skimage.io.imread(img_path)/255.0
    gimg = skimage.util.random_noise(img, 's&p')
    plt.imsave(os.path.join(dest, file_name), gimg)