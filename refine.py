# coding: utf-8


import os
import sys
import argparse
import datetime
import torch
import torchvision
from torchinfo import summary
from trainer import Trainer, GANTrainer, RefineTrainer
from model.layers import MSE_SAMLoss
from model.Lambda import UNet, RefineUNet, Discriminator
from data_loader import PatchMaskDataset
from utils import RandomCrop, RandomHorizontalFlip, RandomRotation
from utils import ModelCheckPoint, Draw_Output
from utils import plot_progress


parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--batch_size', '-b', default=64, type=int, help='Training and validatio batch size')
parser.add_argument('--epochs', '-e', default=150, type=int, help='Train eopch size')
parser.add_argument('--dataset', '-d', default='Harvard', type=str, help='Select dataset')
parser.add_argument('--concat', '-c', default='False', type=str, help='Concat mask by input')
parser.add_argument('--model_name', '-m', default='Lambda', type=str, help='Model Name')
parser.add_argument('--block_num', '-bn', default=9, type=int, help='Model Block Number')
parser.add_argument('--chunck', '-ch', default=2, type=int, help='Mix chuncks')
parser.add_argument('--start_time', '-st', default='0000', type=str, help='start training time')
parser.add_argument('--loss', '-l', default='mse', type=str, help='Loss Mode')
args = parser.parse_args()


# dt_now = datetime.datetime.now()
dt_now = args.start_time
batch_size = args.batch_size
epochs = args.epochs
if args.concat == 'False':
    concat_flag = False
    input_ch = 1
else:
    concat_flag = True
    input_ch = 31
data_name = args.dataset
model_name = args.model_name
block_num = args.block_num
chunck = args.chunck
loss_mode = args.loss


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.benchmark = True


img_path = f'../SCI_dataset/My_{data_name}'
train_path = os.path.join(img_path, 'train_patch_data')
test_path = os.path.join(img_path, 'test_patch_data')
mask_path = os.path.join(img_path, 'mask_data')
callback_path = os.path.join(img_path, 'callback_path')
callback_mask_path = os.path.join(img_path, 'mask_show_data')
callback_result_path = os.path.join('../SCI_result', f'{data_name}_{dt_now}', f'{model_name}_{block_num}')
os.makedirs(callback_result_path, exist_ok=True)
filter_path = os.path.join('../SCI_dataset', 'D700_CSF.mat')
ckpt_path = os.path.join('../SCI_ckpt', f'{data_name}_{dt_now}')
# trained_ckpt_path = f'all_checkpoint_{dt_now.month:02d}{dt_now.day:02d}'
# os.makedirs(trained_ckpt_path, exist_ok=True)
all_trained_ckpt_path = os.path.join(ckpt_path, 'all_trained')
os.makedirs(all_trained_ckpt_path, exist_ok=True)
save_model_name = f'Refine_Stage_{dt_now}'
reconst_name = f'Reconstruct_Stage_{dt_now}'
if os.path.exists(os.path.join(all_trained_ckpt_path, f'{save_model_name}.tar')):
    print('already trained')
    exit(0)




train_transform = (RandomHorizontalFlip(), torchvision.transforms.ToTensor())
test_transform = None
train_dataset = PatchMaskDataset(train_path, mask_path, transform=train_transform, concat=concat_flag)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_dataset = PatchMaskDataset(test_path, mask_path, transform=test_transform, concat=concat_flag)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


model = RefineUNet(31, 31).to(device)
reconst_model = UNet(1, 31)
ckpt = torch.load(os.path.join(all_trained_ckpt_path, f'{reconst_name}.tar'))
reconst_model.load_state_dict(ckpt['model_state_dict'])
reconst_model.to(device)


criterions = {'mse': torch.nn.MSELoss, 'mse_sam': MSE_SAMLoss}
criterion = criterions[loss_mode]()
param = list(model.parameters())
optim = torch.optim.Adam(lr=1e-3, params=param)
scheduler = torch.optim.lr_scheduler.StepLR(optim, 25, .5)

ckpt_cb = ModelCheckPoint(ckpt_path, save_model_name,
                          mkdir=True, partience=1, varbose=True)
trainer = RefineTrainer(model, criterion, optim, reconst_model,
                        callbacks=[ckpt_cb], scheduler=scheduler)
train_loss, val_loss = trainer.train(epochs, train_dataloader, test_dataloader)
torch.save({'model_state_dict': model.state_dict(),
            'optim': optim.state_dict(),
            'train_loss': train_loss, 'val_loss': val_loss,
            'epoch': epochs},
            os.path.join(all_trained_ckpt_path, f'{save_model_name}.tar'))
