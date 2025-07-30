# MPC: Mamba-based Lidar Point Cloud Human Pose Estimation Net

We propose a fast and effective point cloud human pose estimation network, which presents a new idea for human pose estimation. Specifically, by introducing the mamba architecture, we have obtained an MPC network framework that is slightly more accurate than the transformer architecture but with a much faster training speed than transformer. At the same time, we also propose a spatio-temporal coupled heatmap decoder, which can couple spatio-temporal information during the decoding process and improve the model's robustness and spatial perception ability.

## MPC Framework

### Environment

1. Create a conda environment:

   ```bash
   conda create -n mpcbackbone python=3.8
   ```

   Install PyTorch 1.13.1 and CUDA11.7 according to the official instructions.

2. Please go to the mamba official website to query and download causal-conv1d. For more questions about the mamba environment, you can refer to mamba's github page or the following blog: https://blog.csdn.net/leonardotu/article/details/136386581

3. Perform other environment settings through the command:

   ```bash
   accelerate config
   ```

### Dataset

1. **Sloper4d and Lidarhuman26m**
   Please download the Sloper4d and Lidarhuman26m datasets from http://www.lidarhumanmotion.net/data-sloper4d/ and http://www.lidarhumanmotion.net/lidarcap/. Then unzip the dataset to the `./datasets` directory.

2. **SMPL Model**
   Please download the SMPL model (10 shape PCs) from the SMPL official website (https://smpl.is.tue.mpg.de/), then rename the corresponding model and store it in the `smpl_models` path in the following format:

   ```plaintext
   smpl_models
   |-- smpl
   |   |-- SMPL_FEMALE.npz
   |   |-- SMPL_FEMALE.pkl
   |   |-- SMPL_MALE.npz
   |   |-- SMPL_MALE.pkl
   |   |-- SMPL_NEUTRAL.npz
   |   `-- SMPL_NEUTRAL.pkl
   |-- smpl_body_parts_2_faces.json
   `-- smpl_vert_segmentation.json
   ```

3. **Dataset Processing**
   Please use the script we provided to process the dataset, and the processed dataset will be automatically stored in the `./data` path. The specific code is as follows (taking the sloper4d dataset as an example):

   ```bash
   conda activate mpcbackbone
   python -m tools.prepare_sloper4d \
     --raw-path /path/to/raw \
     --buffer-path ./data/lidarh26m
   ```

4. Please check the hyperparameter settings of the relevant files under `./config` and change the necessary paths or names to your actual paths or names.

### Testing

The pre-trained model can be found here:
Please download and place it in:

```plaintext
./work_dir/sloper4d-finetune-MPC/checkpoints/checkpoint_0/model.safetensors
```

Then test on sloper4d:

```bash
accelerate launch main.py configs/sloper4d-finetune-MPC.py --test \
--ckpt work_dir/sloper4d-finetune-MPC/checkpoints/checkpoint_0/model.safetensors
```

### Training

Our model needs to go through a pre-training stage:

```bash
accelerate launch main.py configs/synpretrain-MPC.py
```

After the pre-training is completed, find your pre-trained weight path, which is generally located under `./work_dir/synoretrain-MPC`, and perform training:

```bash
accelerate launch main.py configs/sloper4dfinetune-MPC.py \
--options model.pretrained=work_dir/synpretrain-MPC/checkpoints/checkpoint_0/model.safetensors
```

### Acknowledgements

We would like to thank the following open-source projects for inspiring our work and ideas:

- MambaMos（https://arxiv.org/abs/2404.12794）
- DAPT（https://github.com/AnxQ/dapt/tree/main?tab=readme-ov-file）
- Pointcept（https://github.com/Pointcept/Pointcept）
- Lidarcap（http://www.lidarhumanmotion.net/lidarcap/）
- V2vposenet（https://github.com/dragonbook/V2V-PoseNet-pytorch）
