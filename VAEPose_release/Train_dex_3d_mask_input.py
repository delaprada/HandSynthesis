import os
import torch
import numpy as np
import datetime
from torch.utils.data import DataLoader
from torch.autograd import Variable
from VaePose.config.rhd import RHDConfig as rhgcfg
from VaePose.Model import RGBDecoder, VAE, JointEncoder, JointDecoder, RGBEncoder
from VaePose.utils.tools import create_exp_folder
from VaePose.Loss.loss import mse, kl_div, loss_pck_fn
from VaePose.config.config_dex import Config
from VaePose.DEX_YCB_SF import DEX_YCB_SF
from VaePose.utils.transforms import fetch_transforms
import matplotlib.pyplot as plt

colorlist_pred = ['#660000', '#b30000', '#ff0000', '#ff4d4d', '#ff9999']
colorlist_gt = ['#000066', '#0000b3', '#0000ff', '#4d4dff', '#9999ff']
model_str = ['rgb', '2d', '3d']


def directory_setup(output_folder):
    """
    create folder exp_XXX/ in output_folder and make source_files folder.
    :param output_folder: ./vae_rhd_output
    :return: exp_XXX folder path
    """
    exp_folder = create_exp_folder(output_folder)
    return exp_folder


def plot_fingers(points, plt_specs, c, ax):
    for i in range(5):
        start, end = i*4+1, (i+1)*4+1
        to_plot = np.concatenate((points[start:end], points[0:1]), axis=0)
        ax.plot(to_plot[:,0], to_plot[:,1], to_plot[:,2], plt_specs, color=c[i])
    

# visualize
def save_result_3d(img, target_joint, pred_joint, epoch, iter, joint_type):
    fig = plt.figure(figsize=(15,10))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    ax1.clear()
    ax2.clear()
    ax3.clear()
    
    img = np.transpose(img.cpu(), (1, 2, 0))
    target_joint = target_joint.cpu().numpy()
    pred_joint = pred_joint.cpu().numpy()
    
    # Plot the entire image plus the transformed 2D prediction
    ax1.imshow(img)
    ax1.set_title("Raw image")
    ax1.axis('off')
    
    ax2.plot(target_joint[:, 0], target_joint[:, 1], target_joint[:, 2], 'yo', label='keypoint')

    ax2.plot(target_joint[:5, 0], target_joint[:5, 1],
                target_joint[:5, 2],
                'r',
                label='thumb')

    ax2.plot(target_joint[[0, 5, 6, 7, 8, ], 0], target_joint[[0, 5, 6, 7, 8, ], 1],
                target_joint[[0, 5, 6, 7, 8, ], 2],
                'b',
                label='index')
    ax2.plot(target_joint[[0, 9, 10, 11, 12, ], 0], target_joint[[0, 9, 10, 11, 12], 1],
                target_joint[[0, 9, 10, 11, 12], 2],
                'b',
                label='middle')
    ax2.plot(target_joint[[0, 13, 14, 15, 16], 0], target_joint[[0, 13, 14, 15, 16], 1],
                target_joint[[0, 13, 14, 15, 16], 2],
                'b',
                label='ring')
    ax2.plot(target_joint[[0, 17, 18, 19, 20], 0], target_joint[[0, 17, 18, 19, 20], 1],
                target_joint[[0, 17, 18, 19, 20], 2],
                'b',
                label='pinky')
    # snap convention
    ax2.plot(target_joint[4][0], target_joint[4][1], target_joint[4][2], 'rD', label='thumb')
    ax2.plot(target_joint[8][0], target_joint[8][1], target_joint[8][2], 'r*', label='index')
    ax2.plot(target_joint[12][0], target_joint[12][1], target_joint[12][2], 'rs', label='middle')
    ax2.plot(target_joint[16][0], target_joint[16][1], target_joint[16][2], 'ro', label='ring')
    ax2.plot(target_joint[20][0], target_joint[20][1], target_joint[20][2], 'rv', label='pinky')

    ax2.set_title('GT 3D annotations')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.legend()
    ax2.view_init(-90, -90)
    ax2.axis('off')
    
    ax3.plot(pred_joint[:, 0], pred_joint[:, 1], pred_joint[:, 2], 'yo', label='keypoint')

    ax3.plot(pred_joint[:5, 0], pred_joint[:5, 1],
                pred_joint[:5, 2],
                'r',
                label='thumb')

    ax3.plot(pred_joint[[0, 5, 6, 7, 8, ], 0], pred_joint[[0, 5, 6, 7, 8, ], 1],
                pred_joint[[0, 5, 6, 7, 8, ], 2],
                'b',
                label='index')
    ax3.plot(pred_joint[[0, 9, 10, 11, 12, ], 0], pred_joint[[0, 9, 10, 11, 12], 1],
                pred_joint[[0, 9, 10, 11, 12], 2],
                'b',
                label='middle')
    ax3.plot(pred_joint[[0, 13, 14, 15, 16], 0], pred_joint[[0, 13, 14, 15, 16], 1],
                pred_joint[[0, 13, 14, 15, 16], 2],
                'b',
                label='ring')
    ax3.plot(pred_joint[[0, 17, 18, 19, 20], 0], pred_joint[[0, 17, 18, 19, 20], 1],
                pred_joint[[0, 17, 18, 19, 20], 2],
                'b',
                label='pinky')
    # snap convention
    ax3.plot(pred_joint[4][0], pred_joint[4][1], pred_joint[4][2], 'rD', label='thumb')
    ax3.plot(pred_joint[8][0], pred_joint[8][1], pred_joint[8][2], 'r*', label='index')
    ax3.plot(pred_joint[12][0], pred_joint[12][1], pred_joint[12][2], 'rs', label='middle')
    ax3.plot(pred_joint[16][0], pred_joint[16][1], pred_joint[16][2], 'ro', label='ring')
    ax3.plot(pred_joint[20][0], pred_joint[20][1], pred_joint[20][2], 'rv', label='pinky')

    ax3.set_title('Pred 3D annotations')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax3.legend()
    ax3.view_init(-90, -90)
    ax3.axis('off')
    
    cur_loss = mse(torch.tensor(pred_joint), torch.tensor(target_joint))
    plt.suptitle('mse: ' + str(cur_loss.item()))
    
    plt.show()
    output_dir = "./output_root_relative_original_mask"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(output_dir, 'epoch_' + str(epoch) + '_' + 'iter_' + str(iter) + '_' + joint_type + '.png'))
    plt.close(fig)


def vae_forward_pass(input, target, model, losses, hand_side=None, scale=None,
                     weight=None, bp=True):
    batch_size = input.size(0)
    scale = 1 if scale is None else scale
    recon_batch, mu, logvar = model(input) # [64, 3, 128, 128]
    
    target = target.float() # int64 -> float32
    mse_loss = mse(recon_batch, target, weight)
    kl_loss = kl_div(mu, logvar, input.size(-1), weights=weight, n_joints=rhgcfg.n_joints)
    if bp:
        loss = kl_loss * rhgcfg.kl_term_inc + mse_loss # smaller lambda for kl loss
        loss.backward()
    losses[0] += mse_loss.item() * batch_size
    losses[1] += kl_loss.item() * batch_size
    return kl_loss, mse_loss, recon_batch, mu, logvar


def print_loss(epoch, batch_idx, model_str, losses):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Epoch({epoch}): Batch:{batch_idx}, Time:{now}")
    print(f"\t{model_str[0]}")
    for i in range(len(model_str)):
        ms = model_str[i]
        print(f"{ms}\t{losses[ms][ms]}\t")


def random_mask_3d_joints(data, probability=0.5):    
    shape = data.shape
    mask_2d = np.random.choice([0, 1], size=shape[:-1], p=[probability, 1 - probability])
    mask_3d = torch.tensor(np.repeat(mask_2d[:, :, np.newaxis], shape[-1], axis=-1)).cuda()
    mask_data = mask_3d * data

    return mask_data


def train_rhd(epoch, vaes, ds_loader, optimizer, bp=True):
    vae_rgb_2_rgb, vae_rgb_2_2d, vae_rgb_2_3d, vae_2d_2_rgb, vae_2d_2_2d, vae_2d_2_3d, vae_3d_2_rgb, vae_3d_2_2d, vae_3d_2_3d = vaes
    # model_str = ['rgb', '2d', '3d']
    model_str = ['3d'] # only train 3d pose to 3d pose vae
    
    models = {}
    losses = {}
    for m1 in model_str:
        losses[m1], models[m1] = {}, {}
        for m2 in model_str:
            losses[m1][m2] = [0, 0]  # 0: MSE Loss，1: KL Loss
            vae_str = f'vae_{m1}_2_{m2}'
            eval(vae_str).train()
            models[m1][m2] = eval(vae_str)
    pck_loss = {'pck_10': 0, 'pck_15': 0, 'pck_20': 0}

    for batch_idx, sample in enumerate(ds_loader):
        rgb = sample['img'].float().cuda()
        d2d = sample['joints_img'].float().cuda()
        d3d = sample['joints_coord_cam'].float().cuda()
        
        # data = [rgb, d2d, d3d]
        data = [d3d]

        if rhgcfg.dataset_weighting and ('weight' in sample.keys()):
            weight = sample['weight'].cuda()
        else:
            weight = None
        if rhgcfg.hand_side_invariance and ('hand_side' in sample.keys()):
            hand_side = sample['hand_side'].cuda()
        else:
            hand_side = None
        if rhgcfg.scale_invariance and ('scale' in sample.keys()):
            scale = sample['scale'].cuda()
        else:
            scale = Variable(torch.ones(1)).cuda()

        for i in range(len(model_str)):
            for j in range(len(model_str)):
                # apply random mask for input data
                input_data = random_mask_3d_joints(data[i])
                # input_data = data[i]
                
                mse_loss, kl_loss, recon_batch, mu, logvar = vae_forward_pass(
                    input=input_data, target=data[j], model=models[model_str[i]][model_str[j]],
                    losses=losses[model_str[i]][model_str[j]], hand_side=hand_side, scale=scale,
                    weight=weight, bp=bp)
                
                if bp == True:
                    optimizer.step()
                    optimizer.zero_grad()

                if model_str[i] == 'rgb' and model_str[j] == '3d':
                    pck_loss['pck_10'] += loss_pck_fn(recon_batch.data, d3d, 0.010)
                    pck_loss['pck_15'] += loss_pck_fn(recon_batch.data, d3d, 0.015)
                    pck_loss['pck_20'] += loss_pck_fn(recon_batch.data, d3d, 0.020)
                
                if batch_idx % rhgcfg.interval == 0:
                    save_result_3d(rgb[0], d3d[0], recon_batch.data[0], epoch, batch_idx, '3d')
                    print_loss(epoch, batch_idx, model_str, losses)

    print_loss(epoch, batch_idx, model_str, losses)


def eval_rhd(epoch, vaes, eval_ds_loader):
    print(f"{'='*10} TEST {'='*10}")
    train_rhd(epoch='TEST_'+str(epoch), vaes=vaes, ds_loader=eval_ds_loader, optimizer=None, bp=False)
    print(f"{'='*10}======{'='*10}")


def main():
    # 0. set seeds
    torch.manual_seed(rhgcfg.seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rhgcfg.seed_val)
    np.random.seed(rhgcfg.seed_val)

    # 2. Directory and Logger setup
    exp_folder = directory_setup(rhgcfg.output_folder)
    rhgcfg.exp_folder = exp_folder

    # 3. Load Dataset, Per-processing
    print(f"Joint Dimension: {rhgcfg.dim_joints}")
    
    json_path = os.path.join('./VaePose/config', "cfg.json")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    cfg = Config(json_path).cfg

    # Use GPU if available
    cfg.base.cuda = torch.cuda.is_available()
    if cfg.base.cuda:
        cfg.base.num_gpu = torch.cuda.device_count()
        torch.backends.cudnn.benchmark = True
    gpu_ids = ", ".join(str(i) for i in [j for j in range(cfg.base.num_gpu)])
    print("Using GPU ids: [{}]".format(gpu_ids))
    
    train_transforms, test_transforms = fetch_transforms(cfg)
    # Train dataset
    train_ds = eval(cfg.data.name)(cfg, train_transforms, "train")
    # Val dataset
    if "val" in cfg.data.eval_type:
        validation_ds = eval(cfg.data.name)(cfg, test_transforms, "val")
    # Test dataset
    if "test" in cfg.data.eval_type:
        validation_ds = eval(cfg.data.name)(cfg, test_transforms, "test")
    
    train_ds_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers, pin_memory=True, drop_last=False)
    eval_ds_loader = DataLoader(validation_ds, batch_size=cfg.test.batch_size, shuffle=False, num_workers=cfg.test.num_workers, pin_memory=False, drop_last=False)
    
    # 4. Model [RGB, 2D, 3D]
    ## 4.1 RGB2RGB Model
    # load model from checkpoint
    rgb_encoder = RGBEncoder(z_dim=rhgcfg.z_dim, hand_side_invariance=rhgcfg.hand_side_invariance)
    joint_2d_encoder = JointEncoder(in_dim=[rhgcfg.n_joints, 2], z_dim=rhgcfg.z_dim)
    joint_3d_encoder = JointEncoder(in_dim=[rhgcfg.n_joints, 3], z_dim=rhgcfg.z_dim)
    rgb_decoder = RGBDecoder(z_dim=rhgcfg.z_dim)  # z_dim to image(3,128,128)
    joint_2d_decoder = JointDecoder(z_dim=rhgcfg.z_dim, out_dim=[rhgcfg.n_joints, 2])
    joint_3d_decoder = JointDecoder(z_dim=rhgcfg.z_dim, out_dim=[rhgcfg.n_joints, 3])
    ## 4.2 Structure the encoder-decoder pair
    vae_rgb_2_rgb = VAE(z_dim=rhgcfg.z_dim, encoder=rgb_encoder, decoder=rgb_decoder)
    vae_rgb_2_2d = VAE(z_dim=rhgcfg.z_dim, encoder=rgb_encoder, decoder=joint_2d_decoder)
    vae_rgb_2_3d = VAE(z_dim=rhgcfg.z_dim, encoder=rgb_encoder, decoder=joint_3d_decoder)
    vae_2d_2_rgb = VAE(z_dim=rhgcfg.z_dim, encoder=joint_2d_encoder, decoder=rgb_decoder)
    vae_2d_2_2d = VAE(z_dim=rhgcfg.z_dim, encoder=joint_2d_encoder, decoder=joint_2d_decoder)
    vae_2d_2_3d = VAE(z_dim=rhgcfg.z_dim, encoder=joint_2d_encoder, decoder=joint_3d_decoder)
    vae_3d_2_rgb = VAE(z_dim=rhgcfg.z_dim, encoder=joint_3d_encoder, decoder=rgb_decoder)
    vae_3d_2_2d = VAE(z_dim=rhgcfg.z_dim, encoder=joint_3d_encoder, decoder=joint_2d_decoder)
    vae_3d_2_3d = VAE(z_dim=rhgcfg.z_dim, encoder=joint_3d_encoder, decoder=joint_3d_decoder)
    
    cur_epoch = 0
    
    if cfg.pretrain_model:
        ckpt = torch.load(cfg.pretrain_model)
        cur_epoch = ckpt["epoch"]
        vae_3d_2_3d.load_state_dict(ckpt["vae_3d_2_3d"])
    
    vaes = [vae_rgb_2_rgb, vae_rgb_2_2d, vae_rgb_2_3d,
            vae_2d_2_rgb, vae_2d_2_2d, vae_2d_2_3d,
            vae_3d_2_rgb, vae_3d_2_2d, vae_3d_2_3d]

    vaes_en_de_pair = [rgb_encoder, joint_2d_encoder, joint_3d_encoder, rgb_decoder, joint_2d_decoder,
                       joint_3d_decoder]
    vaes_parameters = []
    ## 5 Drive in GPU
    for vae in vaes_en_de_pair:
        if torch.cuda.is_available():
            vae.cuda()
        vaes_parameters.append({"params": vae.parameters()})
    ## 5. Set optimizer
    optimizer = torch.optim.Adam(vaes_parameters, lr=rhgcfg.lr)
    
    if cfg.pretrain_model:
        optimizer.load_state_dict(ckpt["optimizer"])

    ## 6. Training
    for epoch in range(cur_epoch + 1, rhgcfg.Epochs + 1):
        train_rhd(epoch, vaes, train_ds_loader, optimizer)
        eval_rhd(epoch, vaes, eval_ds_loader)
        
        if epoch % rhgcfg.save_ckpt_interval == 0:
            save_checkpoint(epoch, vaes, optimizer, model_str, exp_folder)


def save_checkpoint(epoch, vaes, optimizer, models, exp_folder):
    vae_rgb_2_rgb, vae_rgb_2_2d, vae_rgb_2_3d, vae_2d_2_rgb, vae_2d_2_2d, vae_2d_2_3d, vae_3d_2_rgb, vae_3d_2_2d, vae_3d_2_3d = vaes
    chech_dic = {
        'epoch': epoch,
        'models': models,
        'optimizer': optimizer.state_dict(),
        'vae_3d_2_3d': vae_3d_2_3d.state_dict(),
    }
    pth_file = os.path.join(exp_folder, f"m-{epoch}-{'-'.join(models)}.pth")
    torch.save(chech_dic, pth_file)


if __name__ == '__main__':
    main()
