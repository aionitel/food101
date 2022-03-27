import numpy as np
import matplotlib.pyplot as plt

# show a couple of images from dataset
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.np()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()