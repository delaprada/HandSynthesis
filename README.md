# Analyzing the Synthetic-to-Real Domain Gap in 3D Hand Pose Estimation

The official code of "Analyzing the Synthetic-to-Real Domain Gap in 3D Hand Pose Estimation. CVPR 2025."

Recent synthetic 3D human datasets for the face, body, and hands have pushed the limits on photorealism. Face recognition and body pose estimation have achieved state-of-the-art performance using synthetic training data alone, but for the hand, there is still a large synthetic-to-real gap. This paper presents the first systematic study of the synthetic-to-real gap of 3D hand pose estimation. We analyze the gap and identify key components such as the forearm, image frequency statistics, hand pose, and object occlusions. To facilitate our analysis, we propose a data synthesis pipeline to synthesize high-quality data. We demonstrate that synthetic hand data can achieve the same level of accuracy as real data when integrating our identified components, paving the path to use synthetic data alone for hand pose estimation.


## Installation

### Install bpy
bpy is used to call different blender API.

Here are steps to install bpy:
1. Create conda environment according to blender python module version (usually python 3.10.0, please refer to the cp version shown in bpy whl below)
    ```
    conda create --name yourEnv python=3.10.0
    ```

2. Download `bpy-3.6.0-cp310-cp310-manylinux_2_28_x86_64.whl` from [Links_for_bpy](https://packagemanager.rstudio.com/pypi/latest/simple/bpy/)

3. Install bpy
    ```
    pip install bpy-3.6.0-cp310-cp310-manylinux_2_28_x86_64.whl
    ```
<br>

### Install blender
blender is the engine to run the rendering script.

Here are steps to install blender:
1. Download [blender Linux](https://www.blender.org/download/release/Blender3.6/blender-3.6.21-linux-x64.tar.xz) from https://www.blender.org/download/lts/3-6/ (Recommend to choose the 3.6 version rather than 4.2)

2. export PATH:
    ```
    export PATH=/home/zhuoran/blender-3.6.X-linux-x64:$PATH
    ```
    Replace "X" with the version you downloaded.

<br>

### Set up the python environment

- [Pytorch3D installation](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).
- [bpycv installation](https://github.com/DIYer22/bpycv?tab=readme-ov-file#-install)

<br>

### Download NIMBLE and MANO models
1. Download the NIMBLE models from [NIMBLE assets](https://drive.google.com/drive/folders/1g7DWuDW5nYI2VDbdemDK2dGVwVHV2a1X?usp=sharing) provided by [NIMBLE](https://github.com/reyuwei/NIMBLE_model).

2. Create an `assets` folder under `./my_NIMBLE_model` and place the downloaded models under `/my_NIMBLE_model/assets/`.

3. Download the MANO_RIGHT.pkl from [MANO](https://mano.is.tue.mpg.de/download.php) in the `Models & Code` section. Place the `MANO_RIGHT.pkl` under `/my_NIMBLE_model/assets/`.

<br>

### Download HDRI scenes

We have uploaded our HDRI scenes for rendering in [HDRI](https://hkustgz-my.sharepoint.com/:u:/g/personal/zzhao074_connect_hkust-gz_edu_cn/ERJY9YB9P95ArV4DFVM5jz8ByEkDL-7ddAtqYwaOYKyQ0A?e=dyDZUc). Download the images and set the `hdri_bg_path` in `config_syn_data.json`.

<br>

### Key files
- `render_syn_data.py`: image rendering script
- `my_NIMBLE_model/main.py`: NIMBLE mesh generation script
- `my_NIMBLE_model/view_samples.ipynb`: image and annotation viewing script

<br>

## Hand data generation

### Run mesh generation script
```
cd my_NIMBLE_model

python main.py
```

<br>

### Run render script

- Linux:
  ```
  /home/zhuoran/blender-3.6.X-linux-x64/blender --background --python render_syn_data.py -- 100 1 0
  ```
  Replace "X" with the version you downloaded.

- MAC (Require installing blender application on MAC first):
  ```
  /Applications/Blender.app/Contents/MacOS/blender --background --python render_syn_data.py -- 100 1 0
  ```

> You can specify your configurations in `config_syn_data.json`.

<br>

## Visualization

![Local Image](./img/syn_data.png)

<br>

Use `view_samples.ipynb` to visualize image and annotation:

![Local Image](./img/syn_data_w_label.png)

<br>

## Citation
If you find our paper useful, please consider citing our paper.

```
@article{zhao2025analyzing,
  title={Analyzing the Synthetic-to-Real Domain Gap in 3D Hand Pose Estimation},
  author={Zhao, Zhuoran and Yang, Linlin and Sun, Pengzhan and Hui, Pan and Yao, Angela},
  journal={arXiv preprint arXiv:2503.19307},
  year={2025}
}
```

## Acknowledgement

Special thanks to these great projects: [NIMBLE](https://github.com/reyuwei/NIMBLE_model?tab=readme-ov-file), [blender-cli-rendering](https://github.com/yuki-koyama/blender-cli-rendering), [bpycv](https://github.com/DIYer22/bpycv).
