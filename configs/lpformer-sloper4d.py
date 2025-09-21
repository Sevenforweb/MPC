_base_ = ["base.py", "base_smpl.py"]

num_epochs = 50
lr = 5e-4
optimizer = dict(type="AdamW", lr=lr, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=lr,
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0
)

model = dict(
    type="LPFormer",  # 关键：模型类型改为LPFormer
    point_dim=3,
    voxel_dim=32,
    emb_dim=256,
    num_heads=8,
    num_layers=4,
    #num_keypoints={{ _base_.keypoint_num }},
    num_keypoints=15,
    bev_dim=512,
    compress_dim=32,
)

train_pipeline = [
    dict(type="Centering"),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomJitter", sigma=0.005, clip=0.02),
    dict(type="RandomFlip", p=0.3, keypoint_flip_index=list(range(15))),
    dict(type="RandomDropout",),
    dict(type="GridSample",
        grid_size=0.01,
        hash_type="fnv",
        mode="train",
        keys=("coord",),
        return_grid_coord=True,
    ),
    dict(type="ShufflePoint",),
    dict(type="ToTensor",),
    dict(type="Collect",
        keys=(
            "coord", 
            "grid_coord",
            "smpl_joints_local",  # LPFormer用这个字段做关键点回归
            "seg_label",          # LPFormer用这个字段做分割
            "vis_label",          # LPFormer用这个字段做可见性
        ),
        feat_keys=("coord",),
    )
]

test_pipeline = [
    dict(type="Centering"),
    dict(type="GridSample",
        grid_size=0.01,
        hash_type="fnv",
        mode="train",
        keys=("coord",),
        return_grid_coord=True,
    ),
    dict(type="ToTensor",),
    dict(type="Collect",
         keys=("coord", "grid_coord", "smpl_joints_local", "seg_label", "vis_label"),
         feat_keys=("coord",),
    )
]

data = dict(
    num_workers=8,
    train_batch_size=8,
    val_batch_size=8,
    train_dataset=dict(
        type='SLOPER4DDataset',
        split='train',
        #keypoint_range={{ _base_.keypoint_range }},
        keypoint_range=[0, 1, 2, 4, 5, 7, 8, 12, 15, 16, 17, 18, 19, 20, 21],  # 直接写死
        transforms=train_pipeline,
        interval=1,
    ),
    val_dataset=dict(
        type='SLOPER4DDataset',
        split='test',
        keypoint_range=[0, 1, 2, 4, 5, 7, 8, 12, 15, 16, 17, 18, 19, 20, 21],  # 直接写死
        transforms=test_pipeline,
        interval=1,
    )
)

exp_name = "{{fileBasenameNoExtension}}"
work_dir = f"./work_dir/{exp_name}"
tracker = dict(
    project_name='pc_pose',
    init_kwargs=dict(
        wandb=dict(name=exp_name)
    )
)