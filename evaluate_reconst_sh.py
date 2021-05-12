# coding: utf-8


import os
import sys
import argparse
import datetime
import torch
from torchsummary import summary
from data_loader import PatchEvalDataset
from model.HSCNN import HSCNN
from model.DeepSSPrior import DeepSSPrior
from model.HyperReconNet import HyperReconNet
from model.Ghost_Mix import GhostMix
from evaluate import RMSEMetrics, PSNRMetrics, SAMMetrics
from evaluate import ReconstEvaluater
from pytorch_ssim import SSIM


parser = argparse.ArgumentParser(description='Evaluate Model')
parser.add_argument('--dataset', '-d', default='Harvard', type=str, help='Select dataset')
parser.add_argument('--concat', '-c', default='False', type=str, help='Concat mask by input')
parser.add_argument('--model_name', '-m', default='HSCNN', type=str, help='Model Name')
parser.add_argument('--block_num', '-b', default=9, type=int, help='Model Block Number')
parser.add_argument('--ratio', '-r', default=2, type=int, help='Ghost ratio')
parser.add_argument('--mode', '-md', default='None', type=str, help='Mix mode')
args = parser.parse_args()


device = 'cpu'
data_name = args.dataset
if args.concat == 'False':
    concat_flag = False
    input_ch = 1
else:
    concat_flag = True
    input_ch = 32


dt_now = datetime.datetime.now()


model_obj = {'HSCNN': HSCNN, 'HyperReconNet': HyperReconNet, 'DeepSSPrior': DeepSSPrior, 'Ghost': GhostMix}
activations = {'HSCNN': 'leaky', 'HyperReconNet': 'relu', 'DeepSSPrior': 'relu', 'Ghost': 'relu'}


model_name = args.model_name
block_num = args.block_num
ratio = args.ratio
mode = args.mode


img_path = f'../SCI_dataset/My_{data_name}'
test_path = os.path.join(img_path, 'eval_data')
mask_path = os.path.join(img_path, 'eval_mask_data')


activation = activations[model_name]
ckpt_dir = f'../SCI_ckpt/{data_name}_0511/all_trained'
if model_name == 'Ghost':
    ckpt_name = f'{model_name}_{activation}_{block_num:02d}_{ratio:02}_{mode}_{dt_now.month:02d}{dt_now.day:02d}'
else:
    ckpt_name = f'{model_name}_{activation}_{block_num:02d}_{dt_now.month:02d}{dt_now.day:02d}'
ckpt_path = os.path.join(ckpt_dir, ckpt_name + '.tar')


output_path = os.path.join('../SCI_result/', data_name, ckpt_name)
output_img_path = os.path.join(output_path, 'output_img')
output_mat_path = os.path.join(output_path, 'output_mat')
output_csv_path = os.path.join(output_path, 'output.csv')
os.makedirs(output_path, exist_ok=True)
os.makedirs(output_img_path, exist_ok=True)
os.makedirs(output_mat_path, exist_ok=True)
print(output_path)
print(output_img_path)
print(output_mat_path)
print(output_csv_path)


test_dataset = PatchEvalDataset(test_path, mask_path, transform=None, concat=concat_flag)
if model_name not in model_obj.keys():
    print('Enter Model Name')
    sys.exit(0)
model = model_obj[model_name](input_ch, 31, block_num=block_num,
                              activation=activations[model_name], 
                              ratio=ratio, mode=mode)


ckpt = torch.load(ckpt_path, map_location=torch.device(device))
model.load_state_dict(ckpt['model_state_dict'])
model.to(device)
# summary(model, (input_ch, 256, 256))
psnr_evaluate = PSNRMetrics().to(device).eval()
ssim_evaluate = SSIM().to(device).eval()
sam_evaluate = SAMMetrics().to(device).eval()
evaluate_fn = [psnr_evaluate, ssim_evaluate, sam_evaluate]


evaluate = ReconstEvaluater(data_name, output_img_path, output_mat_path, output_csv_path)
evaluate.metrics(model, test_dataset, evaluate_fn, ['PSNR', 'SSIM', 'SAM'], hcr=False)
