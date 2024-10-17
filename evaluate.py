import torch
import argparse
from torch.utils.data import DataLoader
import os
import pandas as pd
import time
from dataset import AdvDataset
from model import get_model


def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_model', type=str, default='vit_base_patch16_224',
                        help='the path of adversarial examples.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                        help='input batch size for reference (default: 16)')
    parser.add_argument('--target_model', type=str, default='vit_base_patch16_224', help='')
    parser.add_argument('--name_json', type=str, default='image_name_to_class_id_and_name.json', help='')
    parser.add_argument('--base_path', type=str, default='advimages', help='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Loading dataset
    adv_path = args.base_path + '/model_' + args.source_model + '-method_ATT/'
    dataset = AdvDataset(args.target_model, adv_path, name_to_class_ids_file=args.name_json)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(len(dataset))
    # Loading model
    model = get_model(args.target_model)
    model.cuda()
    model.eval()

    prediction = []
    gts = []
    with torch.no_grad():
        end = time.time()
        for batch_idx, batch_data in enumerate(data_loader):
            batch_x = batch_data[0].cuda()
            batch_y = batch_data[1].cuda()
            batch_name = batch_data[2]
            output = model(batch_x)
            _, pred = output.detach().topk(1, 1, True, True)
            pred = pred.t()
            prediction += list(torch.squeeze(pred.cpu()).numpy())
            gts += list(batch_y.cpu().numpy())
    success_count = 0
    df = pd.DataFrame(columns=['path', 'pre', 'gt'])
    df['path'] = dataset.paths[:len(prediction)]
    df['pre'] = prediction
    df['gt'] = gts

    for i in range(len(df['pre'])):
        if df['pre'][i] != df['gt'][i]:
            success_count += 1
    print("Attack Success Rate for {0} : {1:.1f}%".format(args.target_model, success_count / 1000. * 100))

