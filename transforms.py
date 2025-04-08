import torch
from torchvision import datasets, transforms
from PIL import Image
from typing import List


class MBtransforms():

	def __init__(self, config_dict):

		self.source = config_dict['data']['source']

		self.crop     = transforms.CenterCrop(config_dict['training']['imsize'])
		self.totensor = transforms.ToTensor()
		self.normalise= transforms.Normalize((config_dict['data']['datamean'],), (config_dict['data']['datastd'],))
		self.treatment= config_dict['data']['treatment']

	def transform(self):

		if self.source=='F':

			transform = transforms.Compose([
				self.crop,
				transforms.RandomRotation(360, interpolation=Image.BILINEAR, expand=False),
				self.totensor,
				self.normalise,
				])

		elif self.source=='N':

			transform = transforms.Compose([
				transforms.CenterCrop(18),
				transforms.Resize(150),
				self.crop,
				transforms.RandomRotation(360, interpolation=Image.BILINEAR, expand=False),
				self.totensor,
				self.normalise,
				])

		elif self.source=='FN':

			if self.treatment=='I':
				rotate = transforms.RandomRotation(360, interpolation=Image.BILINEAR, expand=False)
			else:
				angle = float(torch.empty(1).uniform_(0., 360.).item())
				rotate = transforms.RandomRotation([angle, angle], interpolation=Image.BILINEAR, expand=False)
				
			transform_F = transforms.Compose([
					self.crop,
					rotate,
					self.totensor,
					self.normalise,
					])

			transform_N = transforms.Compose([
					transforms.CenterCrop(18),
					transforms.Resize(150),
					self.crop,
					rotate,
					self.totensor,
					self.normalise,
				])

			transform = {'transform_F': transform_F, 'transform_N': transform_N}

		return transform

