import os
import torch
import argparse
from torch import nn
from Normalize import Normalize, TfNormalize
from torch.utils.data import DataLoader
from torch_nets import (
    tf_inception_v3,
    tf_inception_v4,
    tf_resnet_v2_50,
    tf_resnet_v2_101,
    tf_resnet_v2_152,
    tf_inc_res_v2,
    tf_adv_inception_v3,
    tf_ens3_adv_inc_v3,
    tf_ens4_adv_inc_v3,
    tf_ens_adv_inc_res_v2,
    )
from dataset import CNNDataset


def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_model', type=str, default='vit_base_patch16_224', help='')
    parser.add_argument('--batch_size', type=int, default=10, help='')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--name_json', type=str, default='image_name_to_class_id_and_name.json', help='')
    parser.add_argument('--base_path', type=str, default='advimages', help='')
    parser.add_argument('--model_path', type=str, default='./models/', help='')
    args = parser.parse_args()
    return args

def get_model(net_name, model_dir):
    """Load converted model"""
    model_path = os.path.join(model_dir, net_name + '.npy')

    if net_name == 'tf2torch_inception_v3':
        net = tf_inception_v3
    elif net_name == 'tf2torch_inception_v4':
        net = tf_inception_v4
    elif net_name == 'tf2torch_resnet_v2_50':
        net = tf_resnet_v2_50
    elif net_name == 'tf2torch_resnet_v2_101':
        net = tf_resnet_v2_101
    elif net_name == 'tf2torch_resnet_v2_152':
        net = tf_resnet_v2_152
    elif net_name == 'tf2torch_inc_res_v2':
        net = tf_inc_res_v2
    elif net_name == 'tf2torch_adv_inception_v3':
        net = tf_adv_inception_v3
    elif net_name == 'tf2torch_ens3_adv_inc_v3':
        net = tf_ens3_adv_inc_v3
    elif net_name == 'tf2torch_ens4_adv_inc_v3':
        net = tf_ens4_adv_inc_v3
    elif net_name == 'tf2torch_ens_adv_inc_res_v2':
        net = tf_ens_adv_inc_res_v2
    else:
        print('Wrong model name!')

    model = nn.Sequential(
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        TfNormalize('tensorflow'),
        net.KitModel(model_path).eval().cuda(),)
    return model

def verify(model_name, path, adv_path, batch_size, name_to_class_ids_file):

    model = get_model(model_name, path)

    dataset = CNNDataset("inc-v3", adv_path, name_to_class_ids_file=name_to_class_ids_file)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    sum = 0
    for batch_idx, batch_data in enumerate(data_loader):
        batch_x = batch_data[0].cuda()
        batch_y = batch_data[1].cuda()
        batch_name = batch_data[2]

        with torch.no_grad():
            sum += (model(batch_x)[0].argmax(1) != batch_y+1).detach().sum().cpu().item()

    print(model_name + '  acu = {:.1%}'.format(sum / 1000.0))

def main():
    args = arg_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    adv_path = args.base_path + '/model_' + args.source_model + '-method_ATT/'
    batch_size = args.batch_size
    model_names = ['tf2torch_inception_v3','tf2torch_inception_v4','tf2torch_inc_res_v2','tf2torch_resnet_v2_101','tf2torch_ens3_adv_inc_v3','tf2torch_ens4_adv_inc_v3','tf2torch_ens_adv_inc_res_v2']

    models_path = args.model_path
    for model_name in model_names:
        verify(model_name, models_path, adv_path, batch_size, args.name_json)
        print("===================================================")

if __name__ == '__main__':
    main()