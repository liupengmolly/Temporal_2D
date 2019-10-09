import numpy as np
import tqdm
import matplotlib.pyplot as plt
import pickle
import torch
from PIL import Image
from matplotlib.animation import FuncAnimation, writers
import  matplotlib.pyplot as plt
from test import load_model, get_img
from lib.imutils import hp2points
import os
from Config import cfg

parents = [7, 0, 1, 2, 0, 4, 5, 8, 9, 10, 10, 8, 11, 12, 8, 14, 15]

def plt_init(ax, img_h, img_w):
    ax.set_xlim([0, img_w])
    ax.set_ylim([0, img_h])
    ax.invert_yaxis()

    ax.set_title('2d pose')
    ax_2d.append(ax)
    lines_2d.append([])
    ax.set_xlabel('x')
    ax.set_ylabel('y')

def update_video(points):
    global initialized, ax_3d, lines_2d
    if not initialized:
        for i, parent_id in enumerate(parents):
            parent = points[parent_id]
            p = points[i]
            lines_2d[0].append(ax.plot([p[0], parent[0]], [p[1],parent[1]], color='red'))
        initialized = True
    else:
        for i, parent_id in enumerate(parents):
            parent = points[parent_id]
            p = points[i]
            lines_2d[0][i][0].set_xdata([p[0], parent[0]])
            lines_2d[0][i][0].set_ydata([p[1], parent[1]])

def vis(img_paths, img_points):
    fig, ax = plt.subplots(1)

    for id,(img_path, points) in enumerate(zip(img_paths, img_points)):
        im = np.array(Image.open(img_path), dtype=np.uint8)
        ax.imshow(im)
        for i, p in enumerate(points):
            parent = points[parents[i]]
            ax.plot([p[0], parent[0]],[p[1], parent[1]], color='pink')
        id = '0'+str(id) if id<10 else str(id)
        plt.savefig('output/{}.jpg'.format(id))
    plt.close()

def vis_to_video(img_points, img_h, img_w):
    global initialized, lines_2d, ax_2d, ax
    initialized = False
    lines_2d = []
    ax_2d = []

    fig, ax = plt.subplots(1)

    plt_init(ax, img_h, img_w)

    fig.tight_layout()
    fps = 10

    anim = FuncAnimation(fig, update_video, frames=iter(img_points), save_count=len(img_points),
                         interval = 1000/fps, repeat=False)

    writer = writers['ffmpeg']
    writer = writer(fps=fps, metadata={}, bitrate = 30000)
    anim.save('anim_output.mp4', writer=writer)
    plt.close()

def main(model_name):
    # img_pool = '/home/liupeng/workspace/Temporal_2D/data/images/' \
    #            's_09_act_15_subact_02_ca_04/s_09_act_15_subact_02_ca_04_000'
    img_pool = '/home/liupeng/workspace/Temporal_2D/data/video/test.mp4.jpg/'
    start_idx = 70
    cell, hide = None, None

    initial = True
    model = load_model(model_name)
    img_paths = []
    img_points = []
    with torch.no_grad():
        for i in tqdm.tqdm(range(100)):
            img_path = img_pool + '{}.jpg'.format(start_idx + i)
            img, img_h, img_w = get_img(img_path)

            if i > 0:
                initial = False

            outputs, cell, hide = model(img, initial, cell, hide)

            output = outputs[0][0].detach().cpu().numpy()
            points = hp2points(output, cfg, img_h, img_w)

            img_paths.append(img_path)
            img_points.append(points)
            torch.cuda.empty_cache()
    vis_to_video(img_points, img_h, img_w)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    model_name = 'seqlstm_advanced_epoch_16'
    global initialized, lines_2d, ax_2d
    main(model_name)