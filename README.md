# MPC: Mamba-based Lidar Point Cloud Human Pose Estimation Net

我们提出了一种高速有效的点云人体姿态估计网络，该网络呈现了人体姿态估计的一种新思路。具体而言，首先通过引入 mamba 架构，我们得到了一个在精度上略高于 transformer 架构、但训练速率远超 transformer 的 MPC 网络框架。同时，我们还提出了一种时空耦合的热图解码器，能够实现在解码过程中耦合时空信息，提高模型鲁棒性与空间感知能力。

## MPC 框架

### 环境

1. 创建 conda 环境：

   ```bash
   conda create -n mpcbackbone python=3.8
   ```

   按照官方说明安装 PyTorch 1.13.1 和 CUDA11.7

2. 请前往 mamba 官网查询并下载 causal-conv1d。更多有关 mamba 环境问题可参考 mamba 的 github 页面或以下博客：https://blog.csdn.net/leonardotu/article/details/136386581

3. 通过命令进行其它环境设置：

   ```bash
   accelerate config
   ```

### 数据集

1. **Sloper4d 与 Lidarhuman26m**
   请从http://www.lidarhumanmotion.net/data-sloper4d/与http://www.lidarhumanmotion.net/lidarcap/下载 Sloper4d 与 Lidarhuman26m 数据集。然后将数据集解压至`./datasets`目录下。

2. **SMPL 模型**
   请从 SMPL 官网（https://smpl.is.tue.mpg.de/）下载 SMPL 模型（10 shape PCs），然后将对应模型重命名并以如下格式存储至`smpl_models`路径下：

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

3. **数据集处理**
   请采用我们给定的脚本进行数据集的处理，处理好后的数据集将会自动存储至`./data`路径下。具体代码如下（以 sloper4d 数据集为例）：

   ```bash
   conda activate mpcbackbone
   python -m tools.prepare_sloper4d \
     --raw-path /path/to/raw \
     --buffer-path ./data/lidarh26m
   ```

4. 请检查`./config`下有关文件的超参数设置，并将必要路径或名字改为你实际的路径或名字。

### 测试

预训练的模型可以在此处找到：
请下载并放置于：

```plaintext
./work_dir/sloper4d-finetune-MPC/checkpoints/checkpoint_0/model.safetensors
```



然后在 sloper4d 上进行测试：

```bash
accelerate launch main.py configs/sloper4d-finetune-MPC.py --test \
--ckpt work_dir/sloper4d-finetune-MPC/checkpoints/checkpoint_0/model.safetensors
```

### 训练

我们的模型需要经过一个预训练阶段：

```bash
accelerate launch main.py configs/synpretrain-MPC.py
```



预训练完成后找到你的预训练权重路径，一般位于`./work_dir/synoretrain-MPC`下，进行训练：

```bash
accelerate launch main.py configs/sloper4dfinetune-MPC.py \
--options model.pretrained=work_dir/synpretrain-MPC/checkpoints/checkpoint_0/model.safetensors
```

### 感谢

我们感谢下面的开源项目为我们的工作与想法提供了灵感：



- MambaMos（https://arxiv.org/abs/2404.12794）
- DAPT（https://github.com/AnxQ/dapt/tree/main?tab=readme-ov-file）
- Pointcept（https://github.com/Pointcept/Pointcept）
- Lidarcap（http://www.lidarhumanmotion.net/lidarcap/）
- V2vposenet（https://github.com/dragonbook/V2V-PoseNet-pytorch）
