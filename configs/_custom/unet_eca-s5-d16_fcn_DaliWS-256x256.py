_base_ = [
    '../_base_/models/fcn_unet_eca_s5-d16.py', '../_base_/datasets/_custom_DaliWS.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head = dict(
        type='FCNHead',
        num_classes=2),
    auxiliary_head = dict(
        type='FCNHead',
        num_classes=2),
    test_cfg=dict(crop_size=(256, 256), stride=(170, 170)))

train_dataloader = dict(
    batch_size=24,
    num_workers=1)
val_dataloader = dict(
    batch_size=1,
    num_workers=1,)
test_dataloader = val_dataloader

