import visdom
import numpy as np

from torch.utils.data import DataLoader

from VaePose.Loss.loss import mse
from VaePose.Metric.AverageMeter import AverageMeter
from VaePose.Model import RGBEncoder, JointEncoder, RGBDecoder, JointDecoder, VAE
from VaePose.data.RHDDataset import getRHDEvalDataset
from VaePose.utils import View

import argparse
import torch
from VaePose.config.config import get_config


def parse_args():
    parser = argparse.ArgumentParser(description='Test VAEs in RHD.')
    parser.add_argument('--cfg', help='experiment configure file name',
                        default='config/rhd_rgb_abs2d_test.yaml', required=False, type=str)
    args, rest = parser.parse_known_args()
    cfg = get_config(args.cfg)
    return cfg


def main():
    viz = visdom.Visdom()

    cfg = parse_args()
    model_pth = cfg.models_pth
    vaes_dict = torch.load(model_pth)
    print(vaes_dict.keys())

    # Models
    rgb_encoder = RGBEncoder(z_dim=cfg.z_dim, hand_side_invariance=cfg.hand_side_invariance)
    joint_a2d_encoder = JointEncoder(in_dim=[cfg.n_joints, 2], z_dim=cfg.z_dim)
    rgb_decoder = RGBDecoder(z_dim=cfg.z_dim)  # z_dim to image(3,128,128)
    joint_a2d_decoder = JointDecoder(z_dim=cfg.z_dim, out_dim=[cfg.n_joints, 2])
    vae_rgb_2_rgb = VAE(z_dim=cfg.z_dim, encoder=rgb_encoder, decoder=rgb_decoder)
    vae_rgb_2_a2d = VAE(z_dim=cfg.z_dim, encoder=rgb_encoder, decoder=joint_a2d_decoder)
    vae_a2d_2_rgb = VAE(z_dim=cfg.z_dim, encoder=joint_a2d_encoder, decoder=rgb_decoder)
    vae_a2d_2_a2d = VAE(z_dim=cfg.z_dim, encoder=joint_a2d_encoder, decoder=joint_a2d_decoder)

    # Push params
    vae_rgb_2_rgb.load_state_dict(vaes_dict["vae_rgb_2_rgb"])
    vae_rgb_2_a2d.load_state_dict(vaes_dict["vae_rgb_2_a2d"])
    vae_a2d_2_rgb.load_state_dict(vaes_dict["vae_a2d_2_rgb"])
    vae_a2d_2_a2d.load_state_dict(vaes_dict["vae_a2d_2_a2d"])

    # Eval
    eval = AverageMeter()

    # Dataset
    evalDataset = getRHDEvalDataset(cfg.root)
    eval_ds_loader = DataLoader(evalDataset, batch_size=1, num_workers=cfg.works, shuffle=False)
    for sample in eval_ds_loader:
        rgb = sample['rgb']
        rel_2d = sample['rel_2d']
        pred_2d, mean, std = vae_rgb_2_a2d(rgb)
        loss = mse(rel_2d, pred_2d)
        eval.update(loss)
    """
    {'val': tensor(0.0476, grad_fn=<MeanBackward1>),
    'avg': tensor(0.7753, grad_fn=<DivBackward0>),
    'sum': tensor(17.8310, grad_fn=<AddBackward0>), 'count': 23}
    """
    print(eval.avg)

    imagenet_mean = np.array([0.486, 0.459, 0.408])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    for i, sample in enumerate(eval_ds_loader):
        rgb = sample['rgb'][0:1]
        rel_2d = sample['rel_2d'][0:1]
        pred_2d, _, _ = vae_rgb_2_a2d(rgb)

        # viz.image(rgb[0])

        rgb = (np.transpose(rgb[0].cpu().numpy(), (1, 2, 0)) * imagenet_std + imagenet_mean) * 256
        rel_2d = rel_2d.cpu().numpy() * 64 + np.array([128, 128])
        pred_2d = pred_2d.cpu().detach().numpy() * 64 + np.array([128, 128])

        rgb_im = View.view_origin_img(rgb)
        abs_im = View.view_hand_2d(rel_2d.reshape((21, 2)), colorlist=View.colorlist_gt)
        pred_im = View.view_hand_2d(pred_2d.reshape((21, 2)), colorlist=View.colorlist_pred)

        rgb_np = np.array(rgb_im).transpose((2, 0, 1))
        abs_np = np.array(abs_im).transpose((2, 0, 1))
        pred_np = np.array(pred_im).transpose((2, 0, 1))

        viz.images(np.array([rgb_np, abs_np, pred_np]), opts=dict(title=f"{str(i)} : rgb -> abs2d -> pred2d"))

        if i > 100:
            exit()

if __name__ == '__main__':
    main()
