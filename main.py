import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset
from torchvision import datasets, transforms
from torchvision.datasets import MNIST 

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities.model_summary import ModelSummary

from functools import partial
from PIL import Image

import platform

if platform.system()=='Darwin':
    os.environ["GLOO_SOCKET_IFNAME"] = "en0"

quiet = False
torch.set_float32_matmul_precision('medium')

from models import Classifier as CNN
from models import Transformer
from utils import *
from transforms import MBtransforms

from MiraBest import MBFRConfident as MBFull_F
from MiraBest_N import MBFRFull as MBFull_N
from MiraBest_FN import MBFRFull as MBFull_FN

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# extract information from config file:
vars = parse_args()
config_dict, config = parse_config(vars['config'])

train_kwargs = {'batch_size': config_dict['training']['batch_size'],
                'num_workers': config_dict['data']['num_workers'],
                'pin_memory': config_dict['data']['pin_memory'],
                'shuffle': config_dict['data']['shuffle'],
                'persistent_workers': config_dict['data']['persistent_workers']
                }

valid_kwargs = {'num_workers': config_dict['data']['num_workers'],
                'pin_memory': config_dict['data']['pin_memory'],
                'shuffle': False,
                'persistent_workers': config_dict['data']['persistent_workers']
                }

frac_val = config_dict['training']['frac_val']

opt_kwargs  = {'lr': config_dict['training']['lr'],
               'wd': config_dict['training']['decay']
               }

random_state = 10

if torch.cuda.is_available():
    device='cuda'
elif torch.backends.mps.is_built():
    device='mps'
else:
    device='cpu'

# specify source of data:
source = config_dict['data']['source']
if source=='F':
    MBFRFull = MBFull_F
elif source=='N':
    MBFRFull = MBFull_N
elif source=='FN':
    MBFRFull = MBFull_FN

# specify dataset:
dataset = locals()[config_dict['data']['dataset']]

# get transforms for this dataset:
mb_transforms = MBtransforms(config_dict)

# specify model:
modelname = locals()[config_dict['model']['base']]

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():

# -----------------------------------------------------------------------------

    # lightning stuff

    config = {
            'learning_rate': opt_kwargs['lr'],
            'batch_size': train_kwargs['batch_size'],
            'seed': random_state
            }

    # initialise the wandb logger
    wandb_logger = pl.loggers.WandbLogger(project='multiband_transformer', log_model=True, config=config)
    wandb_config = wandb.config

# -----------------------------------------------------------------------------

    # data stuff

    transform = mb_transforms.transform()

    if config_dict['data']['source']=='FN' and config_dict['data']['treatment']=='I':
        train_data_F = MBFull_F('../data', train=True, download=True, transform=transform['transform_F'])
        train_data_N = MBFull_N('../data', train=True, download=True, transform=transform['transform_N'])
        train_data = torch.utils.data.ConcatDataset([train_data_F, train_data_N])

        test_data_F = MBFull_F('../data', train=False, download=True, transform=transform['transform_F'])
        test_data_N = MBFull_N('../data', train=False, download=True, transform=transform['transform_N'])
        test_data = torch.utils.data.ConcatDataset([test_data_F, test_data_N])

    else:
        train_data = dataset('../data', train=True, download=True, transform=transform)
        test_data = dataset('../data', train=False, download=True, transform=transform)


    if frac_val>0.:
        print("Using first {}% of training data for validation.".format(frac_val*100.))

        dataset_size = len(train_data)
        nval = int(frac_val*dataset_size)
        
        indices = list(range(dataset_size))
        train_indices, val_indices = indices[nval:], indices[:nval]

        train_sampler = Subset(train_data, train_indices)
        valid_sampler = Subset(train_data, val_indices)

        train_loader = torch.utils.data.DataLoader(train_sampler, **train_kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_sampler, batch_size=nval, **valid_kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(train_data, **train_kwargs)        
        valid_loader = torch.utils.data.DataLoader(train_data, batch_size=nval, **valid_kwargs)
    

    # witheld test set:
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=False)


# -----------------------------------------------------------------------------

    # model stuff

    # define data handling:
    if config_dict['data']['source']=='FN' and config_dict['data']['treatment']=='C': 
        in_chan=2 
    else: 
        in_chan=1

    # define number of projection heads:
    try:
        ms_projection=config_dict['data']['ms_projection']
    except:
        ms_projection=False

    # call model:
    model = modelname(image_size=config_dict['training']['imsize'], 
                      num_classes=config_dict['training']['num_classes'],
                      in_chan=in_chan,
                      lr=opt_kwargs['lr'],
                      wd=opt_kwargs['wd'],
                      treatment=config_dict['data']['treatment'],
                      ms_projection=ms_projection
                     ).to(device)
    

# -----------------------------------------------------------------------------

    # training stuff

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    trainer = pl.Trainer(max_epochs=config_dict['training']['epochs'],
                         callbacks=[lr_monitor],
                         check_val_every_n_epoch=1,
                         num_sanity_val_steps=0, # 0 : turn off validation sanity check
                         accelerator=device, 
                         devices=1,
                         log_every_n_steps=10,
                         logger=wandb_logger) 

    # train the model
    trainer.fit(model, train_loader, valid_loader)
    
# -----------------------------------------------------------------------------

    # evaluation stuff

    trainer.test(model, test_loader, ckpt_path=None) # test final epoch model

# -----------------------------------------------------------------------------

    # augmented test

    N = len(test_data)
    correct, accuracy = 0, 0

    for i in range(0,N):

        subset_indices = [i] # select your indices here as a list
        subset = torch.utils.data.Subset(test_data, subset_indices)
        testloader_ordered = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)
        data, target = next(iter(testloader_ordered))
        
        samp_c, samp_a = augmented_test(model.to(device), data.to(device), target.to(device), device)

        correct += samp_c
        accuracy += samp_a

    print("Accuracy (1): {}%".format(correct/(N*9)))
    print("Accuracy (2): {}%".format(accuracy/N))

# -----------------------------------------------------------------------------

    wandb.finish()

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    main()