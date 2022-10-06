from __future__ import print_function
from datetime import datetime
# import keras
# from keras.datasets import cifar10
from fastai.data.external import untar_data, URLs
from fastai.vision.all import ImageDataLoaders, RandomResizedCrop, Normalize, imagenet_stats, ToTensor, vision_learner, error_rate, Learner

import torch
from torch.nn import Conv2d, Sequential, Module, ReLU, MaxPool2d, Linear, Flatten

import numpy as np
import sys
import os

import argparse


class GaussianNoise(Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True, device='cuda:0'):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


class RetinaCortexModel(Module):
    def __init__(self, n_retina_layers, n_retina_in_channels, n_retina_out_channels, vvs_width, n_vvs_layers, retina_width, kernel_size, input_shape, out_features, noise_start=0.0, noise_end=0.0):
        super(RetinaCortexModel, self).__init__()

        # RETINA NET
        retina = Sequential(
            # Add Gaussian noise to input
            GaussianNoise(noise_start),
        )

        n_in = n_retina_in_channels
        n_out = n_retina_out_channels if n_retina_layers < 2 else retina_width

        # Add retina layers
        for i in range(n_retina_layers):
            retina.add_module(name=f'retina_{i}', module=Conv2d(in_channels=n_in, out_channels=n_out,
                                                                kernel_size=kernel_size, device='cuda:0', padding='same',
                                                                )
                              )
            n_in = retina_width
            n_out = retina_width if i < n_retina_layers-2 else n_retina_out_channels

        # Add Gaussian noise to output of retina net
        retina.add_module(name='gaussian_noise_out',
                          module=GaussianNoise(noise_end))

        # VVS NET
        n_in = n_out if n_retina_layers > 0 else n_retina_in_channels
        n_out = vvs_width if n_vvs_layers > 0 else n_in
        vvs_net = Sequential()

        # Add VVS layers
        for i in range(n_vvs_layers):
            vvs_net.add_module(name=f'vvs_{i}', module=Conv2d(in_channels=n_in, out_channels=n_out,
                                                              kernel_size=kernel_size, device='cuda:0', padding='same',
                                                              )
                               )
            n_in = n_out

        # Put into learnable layers format
        self.layers = Sequential(
            retina,
            vvs_net,
            Flatten(),
            Linear(
                in_features=input_shape[0]*input_shape[1]*n_out, out_features=out_features)
        )

    def forward(self, x):
        return self.layers(x)

    # def train(self):
    #     pass


def get_parser():
    # Instantiate the parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--trial_label', default='Trial1',
                        help='For labeling different runs of the same model')
    parser.add_argument('--noise_start', type=float, default=0.0,
                        help='Input noise')
    parser.add_argument('--noise_end', type=float, default=0.0,
                        help='Retinal output noise')
    parser.add_argument('--retina_out_weight_reg', type=float, default=0.0,
                        help='L1 regularization on retinal output weights')
    parser.add_argument('--reg', type=float, default=0.0,
                        help='L1 weight regularization for layers besides the retinal output layer')
    parser.add_argument('--retina_hidden_channels', type=int, default=32,
                        help='Channels in hidden layers of retina')
    parser.add_argument('--retina_out_stride', type=int, default=1,
                        help='Stride at output layer of retina')
    parser.add_argument('--task', default='classification',
                        help='e.g. classification or reconstruction')
    parser.add_argument('--kernel_size', type=int, default=9,
                        help='Convolutional filter size')
    parser.add_argument('--retina_layers', type=int, default=2,
                        help='Number of layers in retina')
    parser.add_argument('--n_vvs_layers', type=int, default=2,
                        help='Number of convolutional layers in VVS')
    parser.add_argument('--use_b', type=int, default=1,
                        help='Whether or not to use bias terms in retinal output layer')
    parser.add_argument('--actreg', type=float, default=0.0,
                        help='L1 regularization on retinal output')
    parser.add_argument('--retina_out_width', type=int, default=1,
                        help='Number of output channels in Retina')
    parser.add_argument('--vvs_width', type=int, default=32,
                        help='Number of output channels in VVS layers')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train model')
    parser.add_argument('--dataset', type=str, default="imagenette-320",
                        help='Dataset')

    return parser.parse_args()


if __name__ == "__main__":

    args = get_parser()

    trial_label = args.trial_label
    noise_start = args.noise_start
    noise_end = args.noise_end
    retina_out_weight_reg = args.retina_out_weight_reg
    retina_hidden_channels = args.retina_hidden_channels
    retina_out_stride = args.retina_out_stride
    task = args.task
    kernel_size = args.kernel_size
    retina_layers = args.retina_layers
    n_vvs_layers = args.n_vvs_layers
    use_b = args.use_b
    actreg = args.actreg
    retina_out_width = args.retina_out_width
    vvs_width = args.vvs_width
    epochs = args.epochs
    reg = args.reg
    dataset = args.dataset

    # data_augmentation = True

    save_dir = os.path.join(os.getcwd(), 'saved_models', dataset)
    # model_name = 'cifar10_type_'+trial_label+'_noise_start_'+str(noise_start)+'_noise_end_'+str(noise_end)+'_reg_'+str(reg)+'_retina_reg_'+str(retina_out_weight_reg)+'_retina_hidden_channels_'+str(retina_hidden_channels)+'_SS_'+str(
    #     retina_out_stride)+'_task_'+task+'_kernel_size_'+str(kernel_size)+'_retina_layers_'+str(retina_layers)+'_n_vvs_layers'+str(n_vvs_layers)+'_bias_'+str(use_b)+'_actreg_'+str(actreg)+'_n_retina_out_channels_'+str(retina_out_width)+'_vvs_width_'+str(vvs_width)+'_epochs_'+str(epochs)
    model_name = f"{datetime.now().strftime('%YYYY%mm%dd%HH%MM%SS')}"

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)

    # if use_b == 1:
    #     use_b = True
    # else:
    #     use_b = False

    ##################### Imagenette #####################
    if dataset == 'imagenette-320':
        path = untar_data(URLs.IMAGENETTE_320)
    else:
        print("Dataset not available!")
        exit(1)

    input_shape = (320, 320)
    batch_size = 16
    num_classes = 10


    dls = ImageDataLoaders.from_folder(path, valid='val',
                                       item_tfms=[ToTensor, RandomResizedCrop(size=input_shape)], batch_tfms=Normalize.from_stats(*imagenet_stats), bs=batch_size)
    dls.to('cuda')

    n_retina_layers = 2
    kernel_size = (9, 9)
    n_retina_in_channels = 3    # Number of input channels
    retina_width = 32       # Number of hidden channels if more than 1 retina layer
    n_retina_out_channels = 2    # Number of retina output channels
    vvs_width = 32
    n_vvs_layers = 2

    model = RetinaCortexModel(n_retina_layers=n_retina_layers, kernel_size=kernel_size, n_retina_in_channels=n_retina_in_channels,
                              n_retina_out_channels=n_retina_out_channels, retina_width=retina_width,
                              vvs_width=vvs_width, n_vvs_layers=n_vvs_layers, input_shape=input_shape, out_features=num_classes)

    # print(model)
    learn = Learner(dls, model, metrics=error_rate, path=save_dir, model_dir=model_name)

    # with open('/home/niklas/projects/RetinalResources/saved_models/imagenette-320/2022YYY10m06d12H18M35S', 'rb') as f:
    #     learn = learn.load(file=f)
    
    # print(learn.model)
    # exit(1)

    # learn.fit_one_cycle(epochs, 5e-3)
    learn.fit(n_epoch=epochs)

    with open(model_path, 'wb+') as f:
        learn.save(f)
