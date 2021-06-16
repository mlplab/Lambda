# coding: utf-8


import os
import sys
import argparse
import datetime
import torch
from torchinfo import summary
from data_loader import PatchEvalDataset
from model.Lambda import LambdaNet
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
parser.add_argument('--start_time', '-st', default='0000', type=str, help='start training time')
parser.add_argument('--loss', '-l', default='mse', type=str, help='Loss Mode')
args = parser.parse_args()


device = 'cpu'
dt_now = args.start_time
data_name = args.dataset
if args.concat == 'False':
    concat_flag = False
    input_ch = 1
else:
    concat_flag = True
    input_ch = 32
loss_mode = args.loss


model_name = args.model_name
block_num = args.block_num
ratio = args.ratio
mode = args.mode


img_path = f'../SCI_dataset/My_{data_name}'
test_path = os.path.join(img_path, 'eval_data')
mask_path = os.path.join(img_path, 'eval_mask_data')


ckpt_dir = f'../SCI_ckpt/{data_name}_{dt_now}/all_trained'
ckpt_name = f'Lambda_{dt_now}'
reconst_ckpt = f'Reconstruct_Stage_{dt_now}'
reconst_path = os.path.join(ckpt_dir, f'{reconst_ckpt}.tar')
refine_ckpt = f'Refine_Stage_{dt_now}'
refine_path = os.path.join(ckpt_dir, f'{refine_ckpt}.tar')


output_path = os.path.join('../SCI_result/', f'{data_name}_{dt_now}', ckpt_name)
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
model = LambdaNet(1, 31)
model.load_Reconst(reconst_path)
model.load_Refine(refine_path)
model.to(device)
# summary(model, (input_ch, 256, 256))
psnr_evaluate = PSNRMetrics().to(device).eval()
ssim_evaluate = SSIM().to(device).eval()
sam_evaluate = SAMMetrics().to(device).eval()
evaluate_fn = [psnr_evaluate, ssim_evaluate, sam_evaluate]


evaluate = ReconstEvaluater(data_name, output_img_path, output_mat_path, output_csv_path)
evaluate.metrics(model, test_dataset, evaluate_fn, ['PSNR', 'SSIM', 'SAM'], hcr=False)
