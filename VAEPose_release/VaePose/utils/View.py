import PIL
import numpy as np
import torch
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from VaePose.utils.tools import tensor2np

colorlist_pred = [
    ['#aa0066', '#aa2000', '#aa4000', '#aa8000', '#aaa000'],
    ['#aa0066', '#aa4040', '#aa8040', '#aaa040', '#aae040'],
    ['#aa0066', '#aa0040', '#aa0080', '#aa00a0', '#aa00e0'],
    ['#aa0066', '#aa0040', '#aa0080', '#aa00a0', '#aa00e0'],
    ['#aa0066', '#aa4040', '#aa8080', '#aaa0a0', '#aae0e0'], ]

# 5 根手指，每根手指有 4 个关键点，中心点手心，一共 21 个关键点。
colorlist_gt = [
    ['#000066', '#400000', '#800000', '#a00000', '#e00000'],
    ['#000066', '#004000', '#008000', '#00a000', '#00e000'],
    ['#000066', '#000040', '#000080', '#0000a0', '#0000e0'],
    ['#000066', '#400040', '#800080', '#a000a0', '#e000e0'],
    ['#000066', '#004040', '#008080', '#00a0a0', '#00e0e0'], ]


def view_origin_img(img, show=True, save=True, flag=""):
    if isinstance(img, torch.Tensor):
        if img.dim() == 4:
            img = img.squeeze()
        img = tensor2np(img)
    img = img.astype('uint8')
    if img.shape[0] < 5:
        img = img.transpose(1, 2, 0)

    im = PIL.Image.fromarray(img)
    if show:
        im.show()
    if save:
        im.save(f"{flag}origin.jpg")
    return im


def view_mask(mask, show=True, save=True, flag=""):
    if mask.dim() == 3:
        mask = mask.squeeze()
    im = view_origin_img(mask, show, False)
    if save:
        im.save(f'{flag}mask.jpg')
    return im


def view_hand_2d(points, colorlist=colorlist_pred, im=None, show=False, save=False, flag=""):
    if im == None:
        im = Image.new("RGB", [256, 256], "white")
    draw = ImageDraw.Draw(im)
    if isinstance(points, torch.Tensor):
        if points.dim() == 3:
            points = points.squeeze()
        points = points.cpu().numpy().astype(np.int)
    hand_centor = np.reshape(points[0:1], (1, 2))
    for i in range(5):
        start, end = i * 4 + 1, (i + 1) * 4 + 1
        figures = np.concatenate((points[start:end], hand_centor), axis=0)
        for j in range(4):
            x1, y1, x2, y2 = figures[j, 0], figures[j, 1], figures[j + 1, 0], figures[j + 1, 1]
            draw.line([x1, y1, x2, y2], fill=colorlist[i][j], width=3)
    if show:
        im.show()
    if save:
        im.save(f'{flag}origin_with_hand.jpg')
    return im


def view_hand_3d(points, colorlist=colorlist_gt, show=False, save=False, flag="", echart=False):
    """
    https://www.echartsjs.com/examples/editor.html?c=line3d-orthographic&gl=1
    :param points:
    :param colorlist:
    :param show:
    :param save:
    :param flag:
    :param echart:
    :return:
    """
    if isinstance(points, torch.Tensor):
        if points.dim() == 3:
            points = points.squeeze()
        points = points.cpu().numpy().astype(np.int)
    hand_centor = np.reshape(points[0:1], (1, 3))

    fig = plt.figure(1)
    ax = fig.gca(projection='3d')

    echart_3d_coord = []

    for i in range(5):
        start, end = i * 4 + 1, (i + 1) * 4 + 1
        figures = np.concatenate((points[start: end], hand_centor), axis=0)
        for j in range(4):
            x, y, z = figures[j:j + 2, 0], figures[j:j + 2, 1], figures[j:j + 2, 2]
            ax.plot(x, y, z, c=colorlist[i][j])
            if echart:
                x1, y1, z1 = x[0], y[0], z[0]
                x2, y2, z2 = x[1], y[1], z[1]
                for _ in range(50):
                    p = _ / 50.0
                    echart_3d_coord.append([x1 + p * (x2 - x1), y1 + p * (y2 - y1), z1 + p * (z2 - z1)])
    if save:
        plt.savefig(f"{flag}_hand_3d.png")
    fig.canvas.draw()
    im_np = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    im_np = im_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    im = PIL.Image.fromarray(im_np)
    if show:
        im.show()
    return im, echart_3d_coord
