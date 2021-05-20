# coding: UTF-8


import os
import argparse
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description='Evaluate Model')
parser.add_argument('--dataset', '-d', default='Harvard', type=str, help='Select dataset')

parser.add_argument('--start_time', '-st', default='0000', type=str, help='start training time')
parser.add_argument('--loss', '-l', default='mse', type=str, help='Loss Mode')
args = parser.parse_args()


dt_now = args.start_time
data_name = args.dataset
loss_mode = args.loss


result_dir = f'../SCI_result/{data_name}_{dt_now}'
result_list = os.listdir(result_dir)
csv_list = [dir_name for dir_name in result_list if 'output.csv' in os.listdir(os.path.join(result_dir, dir_name))]


csv_list.sort()
all_result = []
for name in csv_list:
    df = pd.read_csv(os.path.join(result_dir, name, 'output.csv'))
    all_result.append(df.iloc[-1, 1: -1])
all_result = np.array(all_result)
df = pd.DataFrame(all_result, index=csv_list)
df.to_csv(os.path.join(result_dir, 'all_result.csv'))
