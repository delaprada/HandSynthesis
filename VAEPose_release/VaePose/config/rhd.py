class RHDConfig:
    Epochs = 800
    lr = 1e-4
    # lr = 2.5e-4
    bz = 128
    works = 16
    n_joints = 21
    # z_dim = 256
    z_dim = 32

    dim_joints = 3
    both_hands = True
    hand_side_invariance = False
    scale_invariance = True
    rotate_aug = True

    filp_aug = True
    cropped_img = True

    dataset_weighting = False
    crop_size = (256, 256)

    start_epoch = 0

    # Noise
    joint_dropout_prod = 0.0
    save_frequency = 50
    gassian_noise_std = 2.5
    kl_term_reg = 1
    kl_term_inc = 0.01

    test_set_frac = 0.01
    synth_shift_factor = 0.5
    log_interval = 1000

    # seeds
    seed_val = 1

    #Log
    interval = 1
    
    #save ckpt interval
    save_ckpt_interval = 10

    # output folder
    output_folder = './output/vae_dex/'

    exp_folder = None
    
    train_queries = ["trans_images","trans_Ks","joints","scales","trans_joints2d"]
    val_queries = ["trans_images","trans_Ks","joints", "scales", "trans_joints2d"]
    
    base_path = '/root/autodl-tmp/RHD/'
    
    four_channel = False
    
    workers = 4
