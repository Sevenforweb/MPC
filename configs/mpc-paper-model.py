model = dict(
    type="MPCPoseModel",
    num_joints=15,
    grid_size=16,
    coordinate_range=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
    d_model=96,
    d_state=8,
    depth=4,
    kde_sigma=1.25,
    dropout=0.0,
)

loss = dict(
    type="MPCCalibrationLoss",
    heatmap_sigma=0.04,
    heatmap_weight=1.0,
    coordinate_weight=1.0,
    connectivity_weight=0.2,
    foreground_weight=0.2,
    correction_weight=0.01,
)

training = dict(
    optimizer=dict(type="AdamW", lr=2e-4, weight_decay=0.005),
    pretrain_epochs=30,
    finetune_epochs=50,
    finetune_lr=2e-6,
    batch_size=32,
)

paper_module_mapping = dict(
    structured_reconstruction="DensityHilbertSerializer",
    conditional_position_encoding="ConditionalPositionEncoding",
    adaptive_digital_filter="SpatialModulatedSSMBlock",
    physical_space_recoupling="SpatialRecouplingDecoder",
    probabilistic_coordinate_head="MPCPoseModel.heatmap_head",
    calibration_loss="MPCCalibrationLoss",
)
