_base_ = [
    "../_base_/models/deeplabv3plus_r50-d8.py",
    "../_base_/datasets/fib_lco.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_20k.py",
]

train_dataloader = dict(
    sampler=dict(type="DefaultSampler", shuffle=True),
    drop_last=True,
)

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2),
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
