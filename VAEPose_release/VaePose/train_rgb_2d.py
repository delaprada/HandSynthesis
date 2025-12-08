import os

import torch
import numpy as np
import datetime
import logging
from torch.utils.data import DataLoader
from torch.autograd import Variable
from prettytable import PrettyTable

from VaePose.data.RHDDataset import getRHDTrainValDataset
from VaePose.config.rhd import RHDConfig as rhgcfg
from VaePose.Model import RGBDecoder, VAE, JointEncoder, JointDecoder, RGBEncoder
from VaePose.utils.tools import create_exp_folder
from VaePose.utils.tools import is_windows
from VaePose.Loss.loss import mse, kl_div

DEBUG = False

def directory_setup(output_folder, models):
    for model in models:
        output_folder += f'-{model}'
    exp_folder = create_exp_folder(output_folder)
    return exp_folder


def vae_forward_pass(input, target, model, losses, hand_side=None, scale=None, weight=None, bp=True):
    scale = 1 if scale is None else scale
    recon_batch, mu, logvar = model(input)  # [64, 3, 128, 128]
    mse_loss = mse(recon_batch, target, weight)
    kl_loss = kl_div(mu, logvar, input.size(-1), weights=weight, n_joints=rhgcfg.n_joints)
    if bp:
        loss = kl_loss * rhgcfg.kl_term_reg + mse_loss
        loss.backward()
    losses[0] += mse_loss.item()
    losses[1] += kl_loss.item()
    return kl_loss, mse_loss, recon_batch, mu, logvar


def print_emply_loss(epoch, batch_idx, models, losses):
    title = ['']
    for model in models:
        title.append(model)
    table = PrettyTable(title)
    for i in range(len(models)):
        row = [f'{models[i]}']
        for j in range(len(models)):
            mse, kl = losses[models[i]][models[j]]
            row.append(f'{round(mse,7)}, {round(kl, 7)}')
            losses[models[i]][models[j]] = [0.0, 0.0]
        table.add_row(row)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Epoch({epoch}): Batch:{batch_idx}, Time:{now}\n{table}")


def train_rhd(epoch, vaes, ds_loader, optimizer, bp=True):
    vae_rgb_2_rgb, vae_rgb_2_a2d, vae_a2d_2_rgb, vae_a2d_2_a2d = vaes
    vaemodels = {}
    losses = {}
    for m1 in models:
        losses[m1], vaemodels[m1] = {}, {}
        for m2 in models:
            losses[m1][m2] = [0.0, 0.0]  # 0 是 MSE Loss，1 是 KL Loss
            vae_str = f'vae_{m1}_2_{m2}'
            eval(vae_str).train()
            vaemodels[m1][m2] = eval(vae_str)

    for batch_idx, sample in enumerate(ds_loader):
        rgb = sample['rgb'].cuda() if CUDA else sample['rgb']
        d2d = sample['rel_2d'].cuda() if CUDA else sample['rel_2d']
        data = [rgb, d2d]

        if rhgcfg.dataset_weighting and ('weight' in sample.keys()):
            weight = sample['weight'].cuda() if CUDA else sample['weight']
        else:
            weight = None
        if rhgcfg.hand_side_invariance and ('hand_side' in sample.keys()):
            hand_side = sample['hand_side'].cuda() if CUDA else sample['hand_side']
        else:
            hand_side = None
        if rhgcfg.scale_invariance and ('scale' in sample.keys()):
            scale = sample['scale'].cuda() if CUDA else sample['scale']
        else:
            scale = Variable(torch.ones(1)).cuda() if CUDA else Variable(torch.ones(1))

        for i in range(len(models)):
            for j in range(len(models)):
                mi, mj = models[i], models[j]
                mse_loss, kl_loss, recon_batch, mu, logvar = vae_forward_pass(
                    input=data[i], target=data[j], model=vaemodels[mi][mj], losses=losses[mi][mj],
                    hand_side=hand_side, scale=scale, weight=weight, bp=bp)
                if bp == True:
                    optimizer.step()
                    optimizer.zero_grad()
        if batch_idx % rhgcfg.interval == 0:
            print_emply_loss(epoch, batch_idx, models, losses)

        if DEBUG:
            break
    return losses


def eval_rhd(vaes, models, eval_ds_loader):
    vae_rgb_2_rgb, vae_rgb_2_a2d, vae_a2d_2_rgb, vae_a2d_2_a2d = vaes
    eval_model = eval(f"vae_{models[0]}_2_{models[1]}")
    losses = np.array([0.0, 0.0])
    count_loss = np.array([0.0, 0.0, 0.0])

    for idx, sample in enumerate(eval_ds_loader):
        rgb = sample['rgb'].cuda() if CUDA else sample['rgb']
        d2d = sample['rel_2d'].cuda() if CUDA else sample['rel_2d']
        data = [rgb, d2d]

        if rhgcfg.dataset_weighting and ('weight' in sample.keys()):
            weight = sample['weight'].cuda() if CUDA else sample['weight']
        else:
            weight = None
        if rhgcfg.hand_side_invariance and ('hand_side' in sample.keys()):
            hand_side = sample['hand_side'].cuda() if CUDA else sample['hand_side']
        else:
            hand_side = None
        if rhgcfg.scale_invariance and ('scale' in sample.keys()):
            scale = sample['scale'].cuda() if CUDA else sample['scale']
        else:
            scale = Variable(torch.ones(1)).cuda() if CUDA else Variable(torch.ones(1))

        mse_loss, kl_loss, recon_batch, mu, logvar = vae_forward_pass(
            input=rgb, target=d2d, model=eval_model, losses=losses,
            hand_side=hand_side, scale=scale, weight=weight, bp=False)

        if idx % rhgcfg.log_interval:
            count_loss[:2] = count_loss[:2] + losses
            count_loss[2] = idx
            logging.info(f"TEST[{idx}]: {losses / rhgcfg.log_interval}")
            losses = np.array([0.0, 0.0])
    return count_loss

def main():
    # Directory and Logger setup
    exp_folder = directory_setup(rhgcfg.output_folder, models)
    rhgcfg.exp_folder = exp_folder
    init_loging(exp_folder)

    # 3. Load Dataset, Per-processing
    logging.info(f"Models: {models}")
    train_ds, validation_ds = getRHDTrainValDataset(RHDRoot)
    train_ds_loader = DataLoader(train_ds, batch_size=rhgcfg.bz, num_workers=rhgcfg.works, shuffle=True)
    eval_ds_loader = DataLoader(validation_ds, batch_size=rhgcfg.bz, num_workers=rhgcfg.works, shuffle=False)

    # 4. Model
    rgb_encoder = RGBEncoder(z_dim=rhgcfg.z_dim, hand_side_invariance=rhgcfg.hand_side_invariance)
    joint_a2d_encoder = JointEncoder(in_dim=[rhgcfg.n_joints, 2], z_dim=rhgcfg.z_dim)
    rgb_decoder = RGBDecoder(z_dim=rhgcfg.z_dim)  # z_dim to image(3,128,128)
    joint_a2d_decoder = JointDecoder(z_dim=rhgcfg.z_dim, out_dim=[rhgcfg.n_joints, 2])
    vae_rgb_2_rgb = VAE(z_dim=rhgcfg.z_dim, encoder=rgb_encoder, decoder=rgb_decoder)
    vae_rgb_2_a2d = VAE(z_dim=rhgcfg.z_dim, encoder=rgb_encoder, decoder=joint_a2d_decoder)
    vae_a2d_2_rgb = VAE(z_dim=rhgcfg.z_dim, encoder=joint_a2d_encoder, decoder=rgb_decoder)
    vae_a2d_2_a2d = VAE(z_dim=rhgcfg.z_dim, encoder=joint_a2d_encoder, decoder=joint_a2d_decoder)

    vaes = [vae_rgb_2_rgb, vae_rgb_2_a2d, vae_a2d_2_rgb, vae_a2d_2_a2d]

    vaes_en_de_pair = [rgb_encoder, joint_a2d_encoder, rgb_decoder, joint_a2d_decoder]
    vaes_parameters = []
    for vae in vaes_en_de_pair:
        if CUDA:
            vae.cuda()
        vaes_parameters.append({"params": vae.parameters()})
        logging.info(vae)
    ## 5. Set optimizer
    optimizer = torch.optim.Adam(vaes_parameters, lr=rhgcfg.lr)

    ## 6. Training
    for epoch in range(1, rhgcfg.Epochs + 1):
        train_rhd(epoch, vaes, train_ds_loader, optimizer)
        loss = eval_rhd(vaes, models, eval_ds_loader)
        save_checkpoint(epoch, vaes, optimizer, models, loss, exp_folder)


def save_checkpoint(epoch, vaes, optimizer, models, loss, exp_folder):
    vae_rgb_2_rgb, vae_rgb_2_a2d, vae_a2d_2_rgb, vae_a2d_2_a2d = vaes
    chech_dic = {
        'epoch': epoch,
        'models': models,
        'loss': loss,
        'optimizer': optimizer.state_dict(),
        'vae_rgb_2_rgb': vae_rgb_2_rgb.state_dict(),
        'vae_rgb_2_a2d': vae_rgb_2_a2d.state_dict(),
        'vae_a2d_2_rgb': vae_a2d_2_rgb.state_dict(),
        'vae_a2d_2_a2d': vae_a2d_2_a2d.state_dict()}
    pth_file = os.path.join(exp_folder, f"m-{epoch}-{'-'.join(models)}.pth")
    torch.save(chech_dic, pth_file)
    logging.info(f"{loss} with {pth_file}")


def init_loging(exp_folder):
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = os.path.join(exp_folder, f'log_{"_".join(models)}' + '_train.log')
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logging.info(rhgcfg.__dict__)
    logging.info('Start training from [Epoch {}]'.format(rhgcfg.start_epoch))

if __name__ == '__main__':
    CUDA = torch.cuda.is_available()
    models = ['rgb', 'a2d']
    if is_windows() is False:
        RHDRoot = './RHD'
    else:
        RHDRoot = 'E:\\Data\\RHD\\RHD\\'
    main()
