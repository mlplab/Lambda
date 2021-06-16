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
        self.output_progress_path = kwargs.get('output_progress_path')
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

        train_output = np.array(train_output)
        val_output = np.array(val_output)
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
        output = torch.clamp(output,0., 1.)
        labels = torch.clamp(label,0., 1.)
        return [self.psnr(labels, output).item(), self.ssim(labels, output).item(), self.sam(labels, output).item()]


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


class GANTrainer(Trainer):

    def __init__(self, Gmodel, Dmodel, Gcriterion, Dcriterion,
                 Goptim, Doptim, *args, batch_size=64, scheduler=None,
                 callbacks=None, **kwargs):

        self.Gmodel = Gmodel
        self.Gcriterion = Gcriterion
        self.Goptimizer = Goptim
        self.Dmodel = Dmodel
        self.Dcriterion = Dcriterion
        self.Doptimizer = Doptim
        self.scheduler = scheduler
        self.callbacks = callbacks
        self.psnr = PSNRMetrics().eval()
        self.sam = SAMMetrics().eval()
        self.ssim = SSIM().eval()
        shape = kwargs.get('shape', (64, 1))
        self.zeros = torch.zeros(shape).to(device)
        self.ones = torch.ones(shape).to(device)
        self.fake_img_criterion = torch.nn.MSELoss().to(device)
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
            self.Gmodel.train()
            self.Dmodel.train()
            mode = 'Train'
            train_loss = []
            train_Dloss = []
            val_loss = []
            show_train_eval = []
            show_val_eval = []
            desc_str = f'{mode:>5} Epoch: {epoch + 1:05d} / {epochs:05d}'
            with tqdm(train_dataloader, desc=desc_str, ncols=columns, unit='step', ascii=True) as pbar:
                for i, (inputs, labels) in enumerate(pbar):
                    inputs, labels = self._trans_data(inputs, labels)
                    Gloss, output, loss = self._step_G(inputs, labels)
                    Dloss = self._step_D(inputs, labels)
                    train_loss.append(loss.item())
                    show_loss = np.mean(train_loss)
                    train_Dloss.append(Dloss.item())
                    show_Dloss = np.mean(train_Dloss)
                    show_train_eval.append(self._evaluate(output, labels))
                    show_mean = np.mean(show_train_eval, axis=0)
                    evaluate = [f'{show_mean[0]:.5f}', f'{show_mean[1]:.5f}', f'{show_mean[2]:.5f}']
                    self._step_show(pbar, Loss=f'{show_loss:.5f}', DLoss=f'{show_Dloss:.5f}', Evaluate=evaluate)
                    torch.cuda.empty_cache()
            show_mean = np.insert(show_mean, 0, show_loss)
            train_output.append(show_mean)

            mode = 'Val'
            self.Gmodel.eval()
            self.Dmodel.eval()
            desc_str = f'{mode:>5} Epoch: {epoch + 1:05d} / {epochs:05d}'
            with tqdm(val_dataloader, desc=desc_str, ncols=columns, unit='step', ascii=True) as pbar:
                for i, (inputs, labels) in enumerate(pbar):
                    inputs, labels = self._trans_data(inputs, labels)
                    with torch.no_grad():
                        loss, output = self.predict(inputs, labels)
                    val_loss.append(loss.item())
                    show_loss = np.mean(val_loss)
                    show_val_eval.append(self._evaluate(output, labels))
                    show_mean = np.mean(show_val_eval, axis=0)
                    evaluate = [f'{show_mean[0]:.5f}', f'{show_mean[1]:.5f}', f'{show_mean[2]:.5f}']
                    self._step_show(pbar, Loss=f'{show_loss:.5f}', Evaluate=evaluate)
                    torch.cuda.empty_cache()
            show_mean = np.insert(show_mean, 0, show_loss)
            val_output.append(show_mean)
            if self.callbacks:
                for callback in self.callbacks:
                    callback.callback(self.Gmodel, epoch, loss=train_loss,
                                      val_loss=val_loss, save=True, device=device, optim=self.Goptimizer)
            if self.scheduler is not None:
                self.scheduler.step()
            print('-' * int(columns))

        train_output = np.array(train_output)
        val_output = np.array(val_output)
        return train_output, val_output

    def predict(self, inputs, labels):
        output = self.Gmodel(inputs)
        loss = self.fake_img_criterion(output, labels)
        return loss, output

    def _step_G(self, inputs, labels, train=True):
        if train is True:
            self.Goptimizer.zero_grad()
            self.Doptimizer.zero_grad()
        bs = inputs.shape[0]
        fake_img = self.Gmodel(inputs)
        pred_fake = self.Dmodel(fake_img)
        loss = self.Gcriterion(pred_fake, self.ones[:bs])
        show_loss = self.fake_img_criterion(fake_img, labels)
        Gloss = loss + 200 * show_loss
        if train is True:
            Gloss.backward()
            self.Goptimizer.step()
        return Gloss, fake_img, show_loss

    def _step_D(self, inputs, labels, train=True):
        if train is True:
            self.Goptimizer.zero_grad()
            self.Doptimizer.zero_grad()
        bs = inputs.shape[0]
        pred_real = self.Dmodel(labels)
        real_loss = self.Dcriterion(pred_real, self.ones[:bs])
        fake_img = self.Gmodel(inputs)
        pred_fake = self.Dmodel(fake_img)
        fake_loss = self.Dcriterion(pred_fake, self.zeros[:bs])
        if train is True:
            real_loss.backward()
            fake_loss.backward()
            self.Doptimizer.step()
        loss = real_loss + fake_loss
        return loss


class RefineTrainer(Trainer):

    def __init__(self, model, criterion, optimizer, reconst_model,
                 scheduler=None, callbacks=None, **kwargs):
        super().__init__(model, criterion, optimizer, scheduler, callbacks, **kwargs)
        self.reconst_model = reconst_model.eval()

    def _step(self, inputs, labels, train=True):
        if train is True:
            self.optimizer.zero_grad()
        with torch.no_grad():
            inputs = self.reconst_model(inputs)
        output = self.model(inputs)
        loss = self.criterion(output, labels)
        if train is True:
            loss.backward()
            self.optimizer.step()
        return loss, output
