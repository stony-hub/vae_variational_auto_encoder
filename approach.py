import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from network import VAE_network
from dataloader import get_dataloader


class VAE_model(object):
    """
    A Naive Variational Auto Encoder Model
    """
    def __init__(self, load_name, data_root, data_type, img_size, h1_dim, h2_dim, z_dim, cuda: bool):
        self.load_path = os.path.join('model', load_name + '.pth') if load_name is not None else None
        self.data_root = os.path.join('data', data_root)
        self.device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'
        
        self.data_type = data_type  # should be 'MNIST'
        self.img_size = img_size
        self.zdim = z_dim

        self.vae = VAE_network(img_size * img_size, h1_dim, h2_dim, z_dim).to(self.device)
        print('Model initialized')
        self.dataloaders = None
        self.batch_size = None

        if self.load_path is not None and os.path.exists(self.load_path):
            self.vae.load_state_dict(torch.load(self.load_path))
            print('Model loaded')

    def load_data(self, batch_size):
        if self.dataloaders is None or batch_size != self.batch_size:
            self.batch_size = batch_size
            self.dataloaders = get_dataloader(self.data_root, self.data_type, self.img_size, batch_size)
            print('Data loaded')

    def train(self, niter, batch_size, lr, save_name, log_dir=None, start_epoch=0):
        """
        Train the model.
        :param niter: number of epochs to train
        :param batch_size: batch_size
        :param lr: learning rate
        :param save_name: saving path. None means not to save.
        :param log_dir: log_dir of tensorboard. None means not to record.
        :param start_epoch: the id of the beginning epoch
        :return:
        """
        self.load_data(batch_size)
        train_loader, _ = self.dataloaders

        save_path = os.path.join('model', save_name + '.pth') if save_name is not None else None

        optimizer = optim.Adam(self.vae.parameters(), lr)

        writer = SummaryWriter(os.path.join('log', log_dir)) if log_dir is not None else None

        print('Start training from epoch %d to %d' % (start_epoch, niter - 1))

        for epoch in range(start_epoch, niter):

            ##########################################

            epoch_start_time = time.time()
            losses = []
            self.vae.train()
            for i, (x, _) in enumerate(train_loader):
                x = x.view(x.shape[0], -1).to(self.device)
                optimizer.zero_grad()

                re_x, (mu, logv) = self.vae(x)
                loss = self.vae.loss(x, re_x, mu, logv)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            ##########################################

            train_loss = np.array(losses, dtype=np.float).mean()
            test_loss = self.calc_test_loss()
            epoch_time = time.time() - epoch_start_time
            print('Epoch %d finished, time used: %.2f, train_loss: %.2f, test_loss: %.2f'
                  % (epoch, epoch_time, train_loss, test_loss))

            if log_dir is not None:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/test', test_loss, epoch)

        # saving model
        if save_path is not None:
            torch.save(self.vae.state_dict(), save_path)

    def calc_test_loss(self):
        self.vae.eval()
        losses = []
        with torch.no_grad():
            for x, _ in self.dataloaders[1]:
                x = x.view(x.shape[0], -1).to(self.device)
                re_x, (mu, logv) = self.vae(x)
                loss = self.vae.loss(x, re_x, mu, logv)
                losses.append(loss.item())
        test_loss = np.array(losses, dtype=np.float).mean()
        return test_loss

    def test(self, name, number=32):
        """
        Generate some examples.
        :param name: name of the result file
        :param number: number of samples to generate
        """
        path = os.path.join('sample', name + '.png')
        self.vae.eval()
        with torch.no_grad():
            z = torch.randn(number, self.zdim).to(self.device)
            x = self.vae.decode(z)
            x = x.view(-1, 1, self.img_size, self.img_size)
            save_image(x, path)
