import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from transformer import vit_mnist, vit_mb
import utils

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

class VanillaLeNet(nn.Module):
    def __init__(self, in_chan, out_chan, imsize, kernel_size=5):
        super(VanillaLeNet, self).__init__()
        
        z = 0.5*(imsize - 2)
        z = int(0.5*(z - 2))
        
        self.conv1 = nn.Conv2d(in_chan, 6, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size, padding=1)
        self.fc1   = nn.Linear(16*z*z, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, out_chan)
        self.drop  = nn.Dropout(p=0.5)
        
        
    def loss(self,p,y):
        
        # p : softmax(x)
        loss_fnc = nn.NLLLoss()
        loss = loss_fnc(torch.log(p),y)
        
        return loss
     
    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        return
        
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = x.view(x.size()[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
    
        return x

# --------------------------------------------------------------------------

class Classifier(pl.LightningModule):

    """lightning module to reproduce resnet18 baseline"""

    def __init__(self, image_size, num_classes, in_chan, lr, wd, **kwargs):

        super().__init__()
        
        self.model = VanillaLeNet(1, 2, image_size, kernel_size=5)
        self.lr = lr
        self.wd = wd
        self.num_classes = num_classes
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        
        data, target = batch
        output = self.model(data)
        loss = self._loss(target, output)

        self.log(f'train/loss', loss)
        
        return loss

    def _loss(self, targ, pred):

        pred = pred.softmax(dim=1)
        loss_fnc = nn.NLLLoss()
        loss = loss_fnc(torch.log(pred),targ)

        return loss

    def test_step(self, batch, batch_idx):

        data, target = batch
        
        output = self.model(data)
        loss = self._loss(target, output)

        p_y = F.softmax(output, dim=1)
        preds = p_y.argmax(dim=1, keepdim=True)

        correct = preds.eq(target.view_as(preds)).sum().item()
        accuracy = correct / len(target)
        
        self.log(f'test/loss', loss)
        self.log(f'test/accuracy', accuracy)

        return

    def validation_step(self, batch, batch_idx):

        data, target = batch
        output = self.model(data)
        loss = self._loss(target, output)

        p_y = F.softmax(output, dim=1)
        preds = p_y.argmax(dim=1, keepdim=True)

        correct = preds.eq(target.view_as(preds)).sum().item()
        accuracy = correct / len(target)
        
        self.log(f'validation/loss', loss)
        self.log(f'validation/accuracy', accuracy)

        return

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=2, factor=0.9)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'validation/loss'}


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

class Transformer(pl.LightningModule):

    """lightning module for ViT"""

    def __init__(self, image_size, num_classes, in_chan, lr, **kwargs):

        super().__init__()
        
        self.num_classes = num_classes

        self.model = vit_mb(image_size=image_size, num_classes=num_classes, **kwargs)
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        
        data, target = batch
        output = self.model(data)
        loss = self._loss(target, output)

        self.log(f'train/loss', loss)
        
        return loss

    def _loss(self, targ, pred):

        targ_1h = F.one_hot(targ,num_classes=self.num_classes).float() # necessary for MPS backend
        pred = pred.softmax(dim=1)
        loss = F.cross_entropy(targ_1h, pred)
        
        return loss

    def test_step(self, batch, batch_idx):

        data, target = batch
        targ_1h = F.one_hot(target,num_classes=self.num_classes).float() # necessary for MPS backend
        
        output = self.model(data)
        pred = output.softmax(dim=1)

        loss = F.cross_entropy(targ_1h, pred)

        preds = pred.argmax(dim=1, keepdim=True)
        correct = preds.eq(target.view_as(preds)).sum().item()
        accuracy = correct / len(target)
        
        self.log(f'test/loss', loss)
        self.log(f'test/accuracy', accuracy)

        return

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return [optimizer]

# -----------------------------------------------------------------------------
# --------------------------------------------------------------------------