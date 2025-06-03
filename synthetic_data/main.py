from dataset import get_dataset

if __name__ == "__main__":
    syn_frei = get_dataset(
        dat_name="FreiHand",
        base_path="./syn_frei",
        set_name="training",
        queries=["images", "Ks", "joints", "scales", "verts"]
    )
    print('syn_frei dataset length: ', len(syn_frei))
    
    syn_dex = get_dataset(
        dat_name="Dex",
        base_path="./syn_dex",
        set_name="training",
        queries=["images", "Ks", "joints", "scales", "verts"]
    )
    print('syn_dex dataset length: ', len(syn_dex))
    
    syn_obman = get_dataset(
        dat_name="Obman_Syn",
        base_path="./obman",
        set_name="training",
        queries=["images", "Ks", "joints", "scales", "verts"]
    )
    print('syn_obman dataset length: ', len(syn_obman))
    
    syn_nimble = get_dataset(
        dat_name="Random_Nimble",
        base_path="./syn_nimble",
        set_name="training",
        queries=["images", "Ks", "joints", "scales", "verts"]
    )
    print('syn_nimble dataset length: ', len(syn_nimble))

    syn_vae = get_dataset(
        dat_name="VAE_Pose",
        base_path="./syn_vae",
        set_name="training",
        queries=["images", "Ks", "joints", "scales", "verts"]
    )
    print('syn_vae dataset length: ', len(syn_vae))
