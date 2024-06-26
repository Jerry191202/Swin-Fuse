# dataset settings
dataset_type = "FIBDualDataset"
data_root = "../data/"
crop_size = (512, 512)
albu_train_transforms = [dict(type="RandomBrightnessContrast", p=0.5)]
train_pipeline = [
    dict(type="LoadMultipleFIBImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(
        type="RandomResize",
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        resize_type="ResizeMulti",
        keep_ratio=True,
    ),
    dict(type="RandomCropMulti", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlipMulti", prob=0.5),
    dict(type="RandomFlipMulti", prob=0.5, direction="vertical"),
    dict(type="Albu", transforms=albu_train_transforms),
    dict(type="Albu", keymap={"img2": "image"}, transforms=albu_train_transforms),
    dict(type="ConcatCDInput"),
    dict(type="PackSegInputs"),
]
test_pipeline = [
    dict(type="LoadMultipleFIBImageFromFile"),
    dict(type="ResizeMulti", scale=(2048, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="ConcatCDInput"),
    dict(type="PackSegInputs"),
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path="se_img/train",
            img_path2="bse_img/train",
            seg_map_path="masks/train",
        ),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path="se_img/val",
            img_path2="bse_img/val",
            seg_map_path="masks/val",
        ),
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"])
test_evaluator = val_evaluator
