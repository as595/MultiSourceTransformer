import argparse
import configparser as ConfigParser
import ast

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

# ----------------------------------------------------------

def parse_args():
    """
        Parse the command line arguments
        """
    parser = argparse.ArgumentParser()
    parser.add_argument('-C','--config', default="myconfig.txt", required=True, help='Name of the input config file')
    
    args, __ = parser.parse_known_args()
    
    return vars(args)

# -----------------------------------------------------------

def parse_config(filename):
    
    config = ConfigParser.SafeConfigParser(allow_no_value=True)
    config.read(filename)
    
    # Build a nested dictionary with tasknames at the top level
    # and parameter values one level down.
    taskvals = dict()
    for section in config.sections():
        
        if section not in taskvals:
            taskvals[section] = dict()
        
        for option in config.options(section):
            # Evaluate to the right type()
            try:
                taskvals[section][option] = ast.literal_eval(config.get(section, option))
            except (ValueError,SyntaxError):
                err = "Cannot format field '{0}' in config file '{1}'".format(option,filename)
                err += ", which is currently set to {0}. Ensure strings are in 'quotes'.".format(config.get(section, option))
                raise ValueError(err)

    return taskvals, config

# -----------------------------------------------------------

def build_mask(s, margin=2, dtype=torch.float32):
    mask = torch.zeros(1, 1, s, s, dtype=dtype)
    c = (s-1) / 2
    t = (c - margin/100.*c)**2
    sig = 2.
    for x in range(s):
        for y in range(s):
            r = (x - c) ** 2 + (y - c) ** 2
            if r > t:
                mask[..., x, y] = math.exp((t - r)/sig**2)
            else:
                mask[..., x, y] = 1.
    return mask

# -----------------------------------------------------------

def augmented_test(model, data, target, device):

    correct = 0
    rotation_list = range(0, 180, 20)
    
    for r in rotation_list:
        
        # make rotated image:
        rotation_matrix = torch.Tensor([[[np.cos(r/360.0*2*np.pi), -np.sin(r/360.0*2*np.pi), 0],
                                         [np.sin(r/360.0*2*np.pi), np.cos(r/360.0*2*np.pi), 0]]]).to(device)
        grid = F.affine_grid(rotation_matrix, data.size(), align_corners=True)
        data_rotate = F.grid_sample(data, grid, align_corners=True)
        
        # get prediction:
        model.eval()
        x = model(data_rotate)
        
        p_y = F.softmax(x,dim=1)
        preds = p_y.argmax(dim=1, keepdim=True)
        
        correct += preds.eq(target.view_as(preds)).sum().item()
        accuracy = correct / len(rotation_list)

    return correct, accuracy

# -----------------------------------------------------------

