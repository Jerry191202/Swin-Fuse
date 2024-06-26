_base_ = [
    "../_base_/models/upernet_swin.py",
    "../_base_/datasets/fib_lco_dual.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_20k.py",
]

train_dataloader = dict(
    sampler=dict(type="DefaultSampler", shuffle=True),
    drop_last=True,
)

crop_size = (512, 512)
data_preprocessor = dict(
    size=crop_size,
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(in_channels=6),
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2),
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=True, begin=0, end=500),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=1.0,
        begin=500,
        end=40000,
        by_epoch=True,
    ),
]
train_cfg = dict(
    _delete_=True,
    type="EpochBasedTrainLoop",
    max_epochs=40000,
    val_interval=100,
)
default_hooks = dict(
    _delete_=True,
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=True),
    checkpoint=dict(type="CheckpointHook", by_epoch=True, interval=4000),
)

log_processor = dict(by_epoch=True)
