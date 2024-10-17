import torch
import argparse
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from dataset import AdvDataset
import methods


def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--attack', type=str, default='ATT', help='the name of specific attack method')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for reference (default: 16)')
    parser.add_argument('--source_model', type=str, default='vit_base_patch16_224', help='')
    parser.add_argument('--lam', type=float, default=0.01, help='adaptive_factor')
    args = parser.parse_args()
    args.opt_path = os.path.join('./advimages/', 'model_{}-method_{}'.format(args.source_model, args.attack))
    if not os.path.exists(args.opt_path):
        os.makedirs(args.opt_path)
    return args


if __name__ == '__main__':
    args = arg_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # loading dataset
    dataset = AdvDataset(args.source_model, os.path.join('', 'clean_resized_images'), name_to_class_ids_file='image_name_to_class_id_and_name.json')
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(args.attack, args.source_model)

    # Attack
    attack_method = getattr(methods, args.attack)(args.source_model, args)

    # Main
    all_loss_info = {}
    for batch_idx, batch_data in tqdm(enumerate(data_loader)):
        batch_x = batch_data[0]
        batch_y = batch_data[1]
        batch_name = batch_data[3]

        adv_inps, loss_info = attack_method(batch_x, batch_y)
        attack_method._save_images(adv_inps, batch_name, args.opt_path)
