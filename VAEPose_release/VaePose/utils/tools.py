# -*- encoding: utf-8 -*-
# @Time : 2019/2/20 16:26
import datetime
import glob
import os
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt

def create_exp_folder(parent_dir):
    """
    在这个partent_dir文件夹下建立本次训练的输出文件夹。
    :param parent_dir: 某个训练数据集输出的文件夹。
    :return: 返回本次训练的输出文件夹
    """

    exp_nr = 1
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    exp_dir = os.path.join(f'{parent_dir}{now}')
    os.makedirs(exp_dir)
    return exp_dir


def make_folder(directory):
    """
    创建文件夹
    :param directory:
    :return:
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def copy_files_regex(regex_exp, dest_dir):
    """
    将 reges_exp 里面的文件复制到 dest_dir 里面
    :param regex_exp:
    :param dest_dir:
    :return:
    """
    for file in glob.glob(regex_exp):
        shutil.copy(file, dest_dir)


def is_windows():
    import platform
    return 'Windows' in platform.platform()


def tensor2np(tensor):
    """Transforms a pytorch variable/tensor to numpy tensor"""
    if isinstance(tensor, torch.autograd.Variable):
        return tensor.data.cpu().numpy()
    else:
        return tensor.cpu().numpy()

def np2tensor(tensor):
    """Transforms a numpy tensor to pytorch"""
    return torch.from_numpy(tensor).float()


if __name__ == '__main__':
    print(is_windows())