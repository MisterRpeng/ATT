from timm import create_model

MODEL_NAMES = ['vit_base_patch16_224',
               'deit_base_distilled_patch16_224',
               'levit_256',
               'pit_b_224',
               'cait_s24_224',
               'convit_base',
               'tnt_s_patch16_224',
               'visformer_small',
               'vit_large_patch16_224',
               ]


CORR_CKPTS = ['B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz',
              'deit_base_distilled_patch16_224-df68dfff.pth',
              'LeViT-256-13b5763e.pth',
              'pit_b_820.pth',
              'S24_224.pth',
              'convit_base.pth',
              'tnt_s_patch16_224.pth.tar',
              'visformer_small-839e1f5b.pth']


def get_model(model_name):
    if model_name in MODEL_NAMES:
        model = create_model(
            model_name,
            pretrained=True,
            num_classes=1000,
            in_chans=3,
            global_pool=None,
            scriptable=False
        )
    print('Loading Model.')
    return model