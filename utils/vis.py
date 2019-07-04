import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import  matplotlib.pyplot as plt
import os

def vis(img_path, points):
    im = np.array(Image.open(img_path), dtype=np.uint8)
    fig, ax = plt.subplots(1)
    ax.imshow(im)

    parents = [7, 0, 1, 2, 0, 4, 5, 8, 9, 10, 10, 8, 11, 12, 8, 14, 15]
    for i, p in enumerate(points):
        parent = points[parents[i]]
        ax.plot([p[0], parent[0]],[p[1], parent[1]], color='pink')
    plt.show()

