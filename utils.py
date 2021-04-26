# coding: utf-8


import os
import h5py
import pickle
import shutil
import scipy.io
from skimage.transform import rotate
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision


# sns.set(font_scale=2)
sns.set()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ######################### Data Creater ###########################


def normalize(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min())


def calc_filter(img: np.ndarray, filter: np.ndarray) -> np.ndarray:
    return normalize(img.dot(filter)[:, :, ::-1])


def make_patch(data_path: str, save_path: str, size: int=256, step: int=256,
               ch: int=24, data_key: str='data') -> None:

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)

    data_list = os.listdir(data_path)
    data_list.sort()
    for i, name in enumerate(tqdm(data_list, ascii=True)):
        idx = name.split('.')[0]
        f = scipy.io.loadmat(os.path.join(data_path, name))
        data = f[data_key]
        data = normalize(data)
        data = np.expand_dims(np.array(data, np.float32).transpose([2, 0, 1]), axis=0)
        tensor_data = torch.as_tensor(data)
        patch_data = tensor_data.unfold(2, size, step).unfold(3, size, step)
        patch_data = patch_data.permute((0, 2, 3, 1, 4, 5)).reshape(-1, ch, size, size)
        for i in range(patch_data.size()[0]):
            save_data = patch_data[i].to('cpu').detach().numpy().copy().transpose(1, 2, 0)
            save_name = os.path.join(save_path, f'{idx}_{i:05d}.mat')
            scipy.io.savemat(save_name, {'data': save_data})

    return None


def make_patch_h5py(data_path: str, save_path: str, size: int=256, step: int=256,
                    ch: int=24, data_key: str='data') -> None:

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)

    data_list = os.listdir(data_path)
    # data_list.sort()
    for i, name in enumerate(tqdm(data_list, ascii=True)):
        idx = name.split('.')[0]
        # f = scipy.io.loadmat(os.path.join(data_path, name))
        print(os.path.join(data_path, name))
        data = h5py.File(os.path.join(data_path, name), 'r')
        data = np.array(data[data_key].value)
        data = normalize(data)
        data = np.expand_dims(np.array(data, np.float32).transpose([2, 0, 1]), axis=0)[::-1, :, :]
        tensor_data = torch.as_tensor(data)
        patch_data = tensor_data.unfold(2, size, step).unfold(3, size, step)
        patch_data = patch_data.permute((0, 2, 3, 1, 4, 5)).reshape(-1, ch, size, size)
        for i in range(patch_data.size()[0]):
            save_data = patch_data[i].to('cpu').detach().numpy().copy().transpose(1, 2, 0)
            save_name = os.path.join(save_path, f'{idx}_{i:05d}.h5')
            # scipy.io.savemat(save_name, {'data': save_data})
            with h5py.File(save_name, 'w') as f:
                f.create_dataset('data', data=save_data)

    return None


def patch_mask(mask_path: str, save_path: str, size: int=256, step:
               int=256, ch: int=24, data_key: str='data') -> None:

    if os.path.exists(save_path) is True:
        shutil.rmtree(save_path)
    os.mkdir(save_path)

    data = scipy.io.loadmat(mask_path)['data']
    data = np.expand_dims(np.asarray(data, dtype=np.float32).transpose([2, 0, 1]), axis=0)
    tensor_data = torch.as_tensor(data)
    patch_data = tensor_data.unfold(2, size, step).unfold(3, size, step)
    patch_data = patch_data.permute((0, 2, 3, 1, 4, 5)).reshape(-1, ch, size, size)
    for i in tqdm(range(patch_data.size()[0]), ascii=True):
        save_data = patch_data[i].to('cpu').detach().numpy().copy().transpose(1, 2, 0)
        save_name = os.path.join(save_path, f'mask_{i:05d}.mat')
        scipy.io.savemat(save_name, {'data': save_data})

    return None


def patch_mask_h5(mask_path: str, save_path: str, size: int=256, step:
                  int=256, ch: int=24, data_key: str='data') -> None:

    if os.path.exists(save_path) is True:
        shutil.rmtree(save_path)
    os.mkdir(save_path)

    with h5py.File(os.path.join(mask_path), 'r') as f:
        data = np.array(f[data_key].value)
        data = normalize(data)
        data = np.expand_dims(np.array(data, np.float32).transpose([2, 0, 1]), axis=0)[::-1, :, :]
    tensor_data = torch.as_tensor(data)
    patch_data = tensor_data.unfold(2, size, step).unfold(3, size, step)
    patch_data = patch_data.permute((0, 2, 3, 1, 4, 5)).reshape(-1, ch, size, size)
    for i in tqdm(range(patch_data.size()[0]), ascii=True):
        save_data = patch_data[i].to('cpu').detach().numpy().copy().transpose(1, 2, 0)
        save_name = os.path.join(save_path, f'mask_{i:05d}.mat')
        scipy.io.savemat(save_name, {'data': save_data})

    return None


def trans_h52mat(img_path: str) -> np.ndarray:
    f = h5py.File(img_path, 'r')
    bands = np.array(f['bands'], dtype=np.float32)
    rad = np.array(f['rad'], dtype=np.float32).transpose(2, 1, 0)
    rgb = np.array(f['rgb'], dtype=np.float32).transpose(2, 1, 0)
    datas = {'bands': bands, 'data': rad, 'rgb': rgb}
    return datas


def make_icvl_data(img_dir: str, save_dir: str) -> None:
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    img_list = os.listdir(img_dir)
    for name in img_list:
        img_name = name.split('/')[-1]
        datas = trans_h52mat(os.path.join(img_dir, name))
        scipy.io.savemat(os.path.join(save_dir, img_name), datas)

    return None


# ######################### Data Argumentation ###########################


class RandomCrop(object):

    def __init__(self, size: int):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w, _ = img.shape
        i = np.random.randint(0, h - self.size[0], dtype=int)
        j = np.random.randint(0, w - self.size[1], dtype=int)
        return img[i: i + self.size[0], j: j + self.size[1], :].copy()


class RandomHorizontalFlip(object):

    def __init__(self, rate: float=.5):
        self.rate = rate

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.random() < self.rate:
            img = img[:, ::-1, :].copy()
        return img


class RandomRotation(object):

    def __init__(self, angle: list=[0, 90, 180, 270]):
        self.angle = angle

    def __call__(self, img: np.ndarray) -> np.ndarray:
        idx = np.random.randint(len(self.angle))
        img = rotate(img, angle=self.angle[idx])
        return img


# ######################### Callback Layers ###########################


class ModelCheckPoint(object):

    def __init__(self, checkpoint_path: str, model_name: str, mkdir: bool=False,
                 partience: int=1, verbose: bool=True, *args, **kwargs):
        self.checkpoint_path = os.path.join(checkpoint_path, model_name)
        self.model_name = model_name
        self.partience = partience
        self.verbose = verbose
        if mkdir is True:
            if os.path.exists(self.checkpoint_path):
                shutil.rmtree(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.colab2drive_idx = 0
        if 'colab2drive' in kwargs.keys():
            self.colab2drive = kwargs['colab2drive']
            self.colab2drive_path = kwargs['colab2drive_path']
            self.colab2drive_flag = True
        else:
            self.colab2drive_flag = False

    def callback(self, model: str, epoch: int, *args, **kwargs) -> None:
        if 'loss' not in kwargs and 'val_loss' not in kwargs:
            assert 'None Loss'
        else:
            loss = kwargs['loss']
            val_loss = kwargs['val_loss']
        loss = np.mean(loss)
        val_loss = np.mean(val_loss)
        save_file = self.model_name + f'_epoch_{epoch:05d}_loss_{loss:.7f}_valloss_{val_loss:.7f}.tar'
        checkpoint_name = os.path.join(self.checkpoint_path, save_file)

        epoch += 1
        if epoch % self.partience == 0:
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'loss': loss,
                        'val_loss': val_loss,
                        'optim': kwargs['optim'].state_dict()}, checkpoint_name)
            if self.verbose is True:
                print(f'CheckPoint Saved by {checkpoint_name}')
        if self.colab2drive_flag is True and epoch == self.colab2drive[self.colab2drive_idx]:
            colab2drive_path = os.path.join(self.colab2drive_path, save_file)
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'loss': loss,
                        'val_loss': val_loss,
                        'optim': kwargs['optim'].state_dict()}, colab2drive_path)
            self.colab2drive_idx += 1
        return self


class PlotStepLoss(object):

    def __init__(self, checkpoint_path: str, model_name: str, mkdir: bool=False,
                 partience: int=1, verbose: bool=True, *args, **kwargs):
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.partience = partience
        self.verbose = verbose
        if mkdir is True:
            if os.path.exists(self.checkpoint_path):
                shutil.rmtree(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)

    def callback(self, model: str, epoch: int, *args, **kwargs) -> None:
        if 'loss' not in kwargs.keys() and 'val_loss' not in kwargs.keys():
            assert 'None Loss'
        else:
            loss = kwargs['loss']
            val_loss = kwargs['val_loss']
        checkpoint_name = os.path.join(self.checkpoint_path, self.model_name + f'_epoch_{epoch:05d}.png')
        epoch += 1
        if epoch % self.partience == 0:
            plt.figure(figsize=(16, 9))
            plt.plot(loss, marker='.')
            plt.grid()
            plt.xlabel('Step')
            plt.ylabel('MSE')
            plt.savefig(checkpoint_name)
            plt.close()
        return self


class Draw_Output(object):

    def __init__(self, dataset: torch.utils.data.Dataset, data_name: str, *args,
                 save_path: str='output', partience: int=5,
                 verbose: bool=False, ch: int=10, **kwargs):
        '''
        Parameters
        ---
        img_path: str
            image dataset path
        output_data: list
            draw output data path
        save_path: str(default: 'output')
            output img path
        verbose: bool(default: False)
            verbose
        '''
        self.dataset = dataset
        self.data_num = len(self.dataset)
        self.data_name = data_name
        self.save_path = save_path
        self.partience = partience
        self.verbose = verbose
        # self.ch = ch
        # self.filter = np.array(scipy.io.loadmat(filter_path)['T'], dtype=np.float32)
        self.output_ch = {'CAVE': (26, 16, 9), 'Harvard': (21, 11, 12), 'ICVL': (26, 16, 9)}

        ###########################################################
        # Make output directory
        ###########################################################
        if os.path.exists(save_path) is True:
            shutil.rmtree(save_path)
        os.mkdir(save_path)

    def callback(self, model: str, epoch: int, *args, **kwargs) -> None:
        keys = kwargs.keys()
        if epoch % self.partience == 0:
            epoch_save_path = os.path.join(self.save_path, f'epoch_{epoch:05d}')
            os.makedirs(epoch_save_path, exist_ok=True)
            model.eval()
            with torch.no_grad():
                for i, (data, label) in enumerate(self.dataset):
                    data, label = self._trans_data(data, label)

                    output = model(data)

                    diff = torch.abs(output.squeeze() - label.squeeze())
                    diff = diff.to('cpu').detach().numpy().copy()
                    diff = normalize(diff.transpose(1, 2, 0).mean(axis=-1))

                    data = data.squeeze()
                    inputs = normalize(data.to('cpu').detach().numpy().copy())
                    if data.dim() == 3:
                        inputs = inputs[0]
                    inputs = normalize(inputs)
                    outputs = output.squeeze().to('cpu').detach().numpy().copy()
                    outputs = outputs.transpose(1, 2, 0)
                    outputs = normalize(outputs[:, :, self.output_ch[self.data_name]])
                    labels = label.squeeze().to('cpu').detach().numpy().copy()
                    labels = labels.transpose(1, 2, 0)
                    labels = normalize(labels[:, :, self.output_ch[self.data_name]])
                    plt.figure(figsize=(16, 9))
                    self._plot_sub(inputs, 1, title='inputs')
                    self._plot_sub(outputs, 2, title='outputs')
                    self._plot_sub(labels, 3, title='labels')
                    self._plot_sub(diff, 4, title='diff')
                    plt.tight_layout()
                    plt.savefig(os.path.join(epoch_save_path, f'output_{i:05d}.png'), bbox_inches='tight')
                    plt.close()
        return self

    def _trans_data(self, data: torch.Tensor, label: torch.Tensor):
        data = data.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)
        return data, label

    def _plot_sub(self, img: np.ndarray, idx: int, title: str='title') -> None:
        plt.subplot(1, 4, idx)
        if title == 'diff':
            cmap = 'jet'
        elif title == 'inputs':
            cmap = 'gray'
        else:
            cmap = None
        plt.imshow(img, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
        plt.title(title)
        return self


# ######################### Evaluate Tools ###########################


def plot_progress(ckpt_path: str, *args, mode: str='train',
                  eval_names: list=['MSE Loss', 'PSNR', 'SSIM', 'SAM'],
                  **kwargs) -> None:

    figsize = kwargs.get('figsize', (8, 6))
    dir_names = [os.path.join(ckpt_path, name) for name in os.listdir(ckpt_path)
                 if os.path.isdir(os.path.join(ckpt_path, name))]

    for i, eval_name in enumerate(eval_names):
        plt.figure(figsize=figsize)
        for dir_name in dir_names:
            with open(os.path.join(dir_name, f'{mode}.pkl'), 'rb') as f:
                data = pickle.load(f)
            file_name = os.path.split(dir_name)[-1]
            plt.plot(data[:, i], label=f'{file_name}')
        plt.legend()
        plt.tight_layout()
        plt.title(eval_name)
        plt.savefig(os.path.join(ckpt_path, f'{eval_name}_{mode}.png'), bbox_inches='tight')
        plt.close()

