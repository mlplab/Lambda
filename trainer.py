# coding: utf-8


import os
import pickle
import numpy as np
from tqdm import tqdm
from tqdm import tqdm_notebook
from datetime import datetime
from collections import OrderedDict
import torch
# from utils import psnr
from evaluate import PSNRMetrics, SAMMetrics
from pytorch_ssim import SSIM
from utils import normalize


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.benchmark = True
torch.set_printoptions(precision=8)


class Trainer(object):

    def __init__(self, model, criterion, optimizer, scheduler=None, callbacks=None, **kwargs):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callbacks = callbacks
        self.psnr = PSNRMetrics().eval()
        self.sam = SAMMetrics().eval()
        self.ssim = SSIM().eval()
        shape = kwargs.get('shape')
        if shape is None:
            shape = (64, 31, 48, 48)
        self.zeros = torch.zeros(shape).to(device)
        self.ones = torch.ones(shape).to(device)
        self.output_save = kwargs.get('output_save')
        self.output_path = kwargs.get('output_path')
        self.colab_mode = kwargs.get('colab_mode', False)

    def train(self, epochs, train_dataloader, val_dataloader, init_epoch=None):

        if init_epoch is None:
            init_epoch = 0
        elif isinstance(init_epoch, int):
            assert 'Please enter int to init_epochs'

        if self.colab_mode is False:
            _, columns = os.popen('stty size', 'r').read().split()
            columns = int(columns)
        else:
            columns = 200
        train_output = []
        val_output = []
        train_output_loss = []
        val_output_loss = []

        for epoch in range(init_epoch, epochs):
            dt_now = datetime.now()
            print(dt_now)
            self.model.train()
            mode = 'Train'
            train_loss = []
            val_loss = []
            show_train_eval = []
            show_val_eval = []
            desc_str = f'{mode:>5} Epoch: {epoch + 1:05d} / {epochs:05d}'
            with tqdm(train_dataloader, desc=desc_str, ncols=columns, unit='step', ascii=True) as pbar:
                for i, (inputs, labels) in enumerate(pbar):
                    inputs, labels = self._trans_data(inputs, labels)
                    loss, output = self._step(inputs, labels)
                    train_loss.append(loss.item())
                    show_loss = np.mean(train_loss)
                    show_train_eval.append(self._evaluate(output, labels))
                    show_mean = np.mean(show_train_eval, axis=0)
                    evaluate = [f'{show_mean[0]:.7f}', f'{show_mean[1]:.7f}', f'{show_mean[2]:.7f}']
                    self._step_show(pbar, Loss=f'{show_loss:.7f}', Evaluate=evaluate)
                    torch.cuda.empty_cache()
            show_mean = np.insert(show_mean, 0, show_loss)
            train_output.append(show_mean)

            mode = 'Val'
            self.model.eval()
            desc_str = f'{mode:>5} Epoch: {epoch + 1:05d} / {epochs:05d}'
            with tqdm(val_dataloader, desc=desc_str, ncols=columns, unit='step', ascii=True) as pbar:
                for i, (inputs, labels) in enumerate(pbar):
                    inputs, labels = self._trans_data(inputs, labels)
                    with torch.no_grad():
                        loss, output = self._step(inputs, labels, train=False)
                    val_loss.append(loss.item())
                    show_loss = np.mean(val_loss)
                    show_val_eval.append(self._evaluate(output, labels))
                    show_mean = np.mean(show_val_eval, axis=0)
                    evaluate = [f'{show_mean[0]:.7f}', f'{show_mean[1]:.7f}', f'{show_mean[2]:.7f}']
                    self._step_show(pbar, Loss=f'{show_loss:.7f}', Evaluate=evaluate)
                    torch.cuda.empty_cache()
            show_mean = np.insert(show_mean, 0, show_loss)
            val_output.append(show_mean)
            if self.callbacks:
                for callback in self.callbacks:
                    callback.callback(self.model, epoch, loss=train_loss,
                                      val_loss=val_loss, save=True, device=device, optim=self.optimizer)
            if self.scheduler is not None:
                self.scheduler.step()
            print('-' * int(columns))

        return train_output, val_output

    def _trans_data(self, inputs, labels):
        inputs = inputs.to(device)
        labels = labels.to(device)
        return inputs, labels

    def _step(self, inputs, labels, train=True):
        if train is True:
            self.optimizer.zero_grad()
        output = self.model(inputs)
        loss = self.criterion(output, labels)
        if train is True:
            loss.backward()
            self.optimizer.step()
        return loss, output

    def _step_show(self, pbar, *args, **kwargs):
        if device == 'cuda':
            kwargs['Allocate'] = f'{torch.cuda.memory_allocated(0) / 1024 ** 3:.3f}GB'
            kwargs['Cache'] = f'{torch.cuda.memory_cached(0) / 1024 ** 3:.3f}GB'
        pbar.set_postfix(kwargs)
        return self

    def _evaluate(self, output, label):
        output = self._cut(output)
        labels = self._cut(label)
        return [self.psnr(labels, output).item(), self.ssim(labels, output).item(), self.sam(labels, output).item()]

    def _cut(self, x):
        bs, _, _, _ = x.size()
        x = torch.where(x > 1., self.ones[:bs], x)
        x = torch.where(x < 0., self.zeros[:bs], x)
        return x


class Deeper_Trainer(Trainer):

    def _step(self, inputs, labels, train=True):
        if train is True:
            self.optimizer.zero_grad()
        output_6, output_12, output = self.model(inputs)
        labels_6 = labels[:, ::4]
        labels_12 = labels[:, ::2]
        loss = .1 * self.criterion(output_6, labels_6) + .1 * self.criterion(output_12, labels_12) + self.criterion(output, labels)
        if train is True:
            loss.backward()
            self.optimizer.step()
        show_loss = torch.nn.functional.mse_loss(output, labels)
        return show_loss
