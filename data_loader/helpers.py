import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.np()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()