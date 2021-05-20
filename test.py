# coding: UTF-8


import os


ckpt_dir = '../SCI_ckpt/CAVE_0520/all_trained'
ckpt_list = os.listdir(ckpt_dir)


for name in ckpt_list:
    print(name)
    name_split = name.split('.')[0].split('_')
    name_split.remove('0519')
    name_after = ('_').join(name_split) + '.tar'
    os.rename(os.path.join(ckpt_dir, name), os.path.join(ckpt_dir, name_after))
