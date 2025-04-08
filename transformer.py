import torch
import torch.nn as nn

import warnings
from collections import OrderedDict
from typing import Callable, List, Optional, Sequence, Tuple, Union, Any, NamedTuple
from functools import partial
import math

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities.model_summary import ModelSummary

from torchvision.models.vision_transformer import VisionTransformer, Encoder, EncoderBlock, ConvStemConfig


# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------

class MSEncoder(nn.Module):
	"""Transformer Model Encoder for sequence to sequence translation."""

	def __init__(
		self,
		seq_length: int,
		num_layers: int,
		num_heads: int,
		hidden_dim: int,
		mlp_dim: int,
		dropout: float,
		attention_dropout: float,
		norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
	):
		super().__init__()

		self.seq_length = seq_length
		self.n_patch = seq_length - 1

		# Note that batch_size is on the first dim because
		# we have batch_first=True in nn.MultiAttention() by default
		self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # position + cls token encoding
		self.src_embedding = nn.Parameter(torch.empty(1, 3, hidden_dim).normal_(std=0.02))  # source encoding: 2 sources + class token

		self.dropout = nn.Dropout(dropout)

		layers: OrderedDict[str, nn.Module] = OrderedDict()
		for i in range(num_layers):
			layers[f"encoder_layer_{i}"] = EncoderBlock(
				num_heads,
				hidden_dim,
				mlp_dim,
				dropout,
				attention_dropout,
				norm_layer,
			)
		self.layers = nn.Sequential(layers)
		self.ln = norm_layer(hidden_dim)

	def forward(self, input: torch.Tensor):

		torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")

		x0 = torch.split(input, [1, self.n_patch, self.n_patch], dim=1)[0] + self.pos_embedding[:,0,:]  + self.src_embedding[:,0,:] 
		x1 = torch.split(input, [1, self.n_patch, self.n_patch], dim=1)[1] + self.pos_embedding[:,1:,:] + self.src_embedding[:,1,:]
		x2 = torch.split(input, [1, self.n_patch, self.n_patch], dim=1)[2] + self.pos_embedding[:,1:,:] + self.src_embedding[:,2,:]
		x = torch.cat((x0,x1,x2), dim=1)
				
		x = self.dropout(x)
		x = self.layers(x)
		x = self.ln(x)

		return self.ln(self.layers(self.dropout(input)))


# --------------------------------------------------------------------------------------------------------

class VisionTransformer1C(VisionTransformer):
	"""Vision Transformer as per https://arxiv.org/abs/2010.11929."""

	def __init__(
		self,
		image_size: int,
		patch_size: int,
		num_layers: int,
		num_heads: int,
		hidden_dim: int,
		mlp_dim: int,
		dropout: float = 0.0,
		attention_dropout: float = 0.0,
		num_classes: int = 1000, 
		representation_size: Optional[int] = None,
		norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
		conv_stem_configs: Optional[List[ConvStemConfig]] = None,
	):
		super().__init__(image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, num_classes, representation_size)

		# projection head for a 1 channel input:
		self.conv_proj = nn.Conv2d(in_channels=1, 
								   out_channels=hidden_dim, 
								   kernel_size=patch_size, 
								   stride=patch_size
								   )

# --------------------------------------------------------------------------------------------------------

class VisionTransformer2C(VisionTransformer):
	"""Vision Transformer as per https://arxiv.org/abs/2010.11929."""

	def __init__(
		self,
		image_size: int,
		patch_size: int,
		num_layers: int,
		num_heads: int,
		hidden_dim: int,
		mlp_dim: int,
		dropout: float = 0.0,
		attention_dropout: float = 0.0,
		num_classes: int = 1000, 
		representation_size: Optional[int] = None,
		norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
		conv_stem_configs: Optional[List[ConvStemConfig]] = None,
	):
		super().__init__(image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, num_classes, representation_size)

		# projection head for a 1 channel input:
		self.conv_proj = nn.Conv2d(in_channels=2, 
								   out_channels=hidden_dim, 
								   kernel_size=patch_size, 
								   stride=patch_size
								   )



# --------------------------------------------------------------------------------------------------------

class MultiSourceVisionTransformer(VisionTransformer):
	"""Vision Transformer as per https://arxiv.org/abs/2010.11929."""

	def __init__(
		self,
		image_size: int,
		patch_size: int,
		num_layers: int,
		num_heads: int,
		hidden_dim: int,
		mlp_dim: int,
		dropout: float = 0.0,
		attention_dropout: float = 0.0,
		num_classes: int = 1000, 
		representation_size: Optional[int] = None,
		norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
		conv_stem_configs: Optional[List[ConvStemConfig]] = None,
		ms_projection: Optional = True,
	):
		super().__init__(image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, num_classes, representation_size)

		self.ms_projection = ms_projection

		
		if self.ms_projection: 

			# projection heads for 2 separate sources:
			self.conv_proj1 = nn.Conv2d(in_channels=1, 
									   out_channels=hidden_dim, 
									   kernel_size=patch_size, 
									   stride=patch_size
									   )

			self.conv_proj2 = nn.Conv2d(in_channels=1, 
									   out_channels=hidden_dim, 
									   kernel_size=patch_size, 
									   stride=patch_size
									   )
		else:

			# single projection head:
			self.conv_proj = nn.Conv2d(in_channels=1, 
									   out_channels=hidden_dim, 
									   kernel_size=patch_size, 
									   stride=patch_size
									   )


		seq_length = (image_size // patch_size) ** 2

		# Add a class token
		self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
		seq_length += 1

		# encoder for multiple sources:
		self.encoder = MSEncoder(seq_length,
								 num_layers,
								 num_heads,
								 hidden_dim,
								 mlp_dim,
								 dropout,
								 attention_dropout,
								 norm_layer,
								)

		self.seq_length = seq_length

		# classification head:
		heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
		if representation_size is None:
			heads_layers["dropout"] = nn.Dropout(0.1)
			heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
		else:
			heads_layers["pre_logits_1"] = nn.Linear(hidden_dim, 120)
			heads_layers["act_1"] = nn.ReLU()
			heads_layers["pre_logits_2"] = nn.Linear(120, 84)
			heads_layers["act_2"] = nn.ReLU()
			heads_layers["head"] = nn.Linear(84, num_classes)
			
		self.heads = nn.Sequential(heads_layers)

		if isinstance(self.conv_proj, nn.Conv2d):
			# Init the patchify stem
			fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
			nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
			if self.conv_proj.bias is not None:
				nn.init.zeros_(self.conv_proj.bias)
		elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
			# Init the last 1x1 conv of the conv stem
			nn.init.normal_(
				self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
			)
			if self.conv_proj.conv_last.bias is not None:
				nn.init.zeros_(self.conv_proj.conv_last.bias)

		if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
			fan_in = self.heads.pre_logits.in_features
			nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
			nn.init.zeros_(self.heads.pre_logits.bias)

		if isinstance(self.heads.head, nn.Linear):
			nn.init.zeros_(self.heads.head.weight)
			nn.init.zeros_(self.heads.head.bias)


	def process_input(self, x: torch.Tensor) -> torch.Tensor:
		return self._process_input(x)

	def _process_input(self, x: torch.Tensor) -> torch.Tensor:
		
		n, c, h, w = x.shape
		p = self.patch_size

		torch._assert(c == 2, f"Only one source included! Expected 2 but got {c}!")
		torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
		torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
		n_h = h // p
		n_w = w // p

		# projection per source
		if self.ms_projection: 
			# (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
			x1 = self.conv_proj1(x[:,0,:,:].unsqueeze(1))
			x2 = self.conv_proj2(x[:,1,:,:].unsqueeze(1))

			# (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
			x1 = x1.reshape(n, self.hidden_dim, n_h * n_w)
			x2 = x2.reshape(n, self.hidden_dim, n_h * n_w)

			# (n, hidden_dim, (n_h * n_w)) & (n, hidden_dim, (n_h * n_w)) -> (n, hidden_dim, 2*(n_h * n_w))
			x = torch.cat((x1,x2), dim=2)

		# common projection across sources
		else: 
			# (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
			x1 = self.conv_proj(x[:,0,:,:].unsqueeze(1))
			x2 = self.conv_proj(x[:,1,:,:].unsqueeze(1))

			# (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
			x1 = x1.reshape(n, self.hidden_dim, n_h * n_w)
			x2 = x2.reshape(n, self.hidden_dim, n_h * n_w)

			# (n, hidden_dim, (n_h * n_w)) & (n, hidden_dim, (n_h * n_w)) -> (n, hidden_dim, 2*(n_h * n_w))
			x = torch.cat((x1,x2), dim=2)


		# (n, hidden_dim, 2 * (n_h * n_w)) -> (n, 2 * (n_h * n_w), hidden_dim)
		# The self attention layer expects inputs in the format (N, S, E)
		# where S is the source sequence length, N is the batch size, E is the
		# embedding dimension
		x = x.permute(0, 2, 1)

		return x

	def forward(self, x: torch.Tensor):

		# Reshape and permute the input tensor
		x = self.process_input(x)
		n = x.shape[0]

		# Expand the class token to the full batch
		batch_class_token = self.class_token.expand(n, -1, -1)
		x = torch.cat([batch_class_token, x], dim=1)

		x = self.encoder(x)

		# Classifier "token" as used by standard language architectures
		x = x[:, 0]
		
		x = self.heads(x)

		return x
		
		
# --------------------------------------------------------------------------------------------------------

def _vision_transformer(
	patch_size: int,
	num_layers: int,
	num_heads: int,
	hidden_dim: int,
	mlp_dim: int,
	representation_size: Optional,
	weights: Optional,
	progress: bool,
	**kwargs: Any,
	):
	
	image_size = kwargs.pop("image_size")
	num_classes = kwargs.pop("num_classes")

	treatment = kwargs.pop("treatment")
	ms_projection = kwargs.pop("ms_projection")
	
	if treatment=='S':
		model = MultiSourceVisionTransformer( # multi-source vision transformer w/ separate source projections
			image_size=image_size,
			patch_size=patch_size,
			num_layers=num_layers,
			num_heads=num_heads,
			hidden_dim=hidden_dim,
			mlp_dim=mlp_dim,
			num_classes=num_classes,
			representation_size=representation_size,
			**kwargs,
		)
	elif treatment=='I':
		model = VisionTransformer1C( # standard vision transformer w/ 1 channel input
			image_size=image_size,
			patch_size=patch_size,
			num_layers=num_layers,
			num_heads=num_heads,
			hidden_dim=hidden_dim,
			mlp_dim=mlp_dim,
			num_classes=num_classes,
	#        representation_size=representation_size,
			**kwargs,
		)
	elif treatment=='C':
		model = VisionTransformer2C( # standard vision transformer w/ 2 channel input
			image_size=image_size,
			patch_size=patch_size,
			num_layers=num_layers,
			num_heads=num_heads,
			hidden_dim=hidden_dim,
			mlp_dim=mlp_dim,
			num_classes=num_classes,
	#        representation_size=representation_size,
			**kwargs,
		)

	if weights:
		model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

	return model

# --------------------------------------------------------------------------------------------------------

def vit_mnist(*, weights: Optional = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
	"""
	Constructs a vit_b_16 architecture from
	`An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

	Args:
		weights (:class:`~torchvision.models.ViT_B_16_Weights`, optional): The pretrained
			weights to use. See :class:`~torchvision.models.ViT_B_16_Weights`
			below for more details and possible values. By default, no pre-trained weights are used.
		progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
		**kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
			base class. Please refer to the `source code
			<https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
			for more details about this class.

	.. autoclass:: torchvision.models.ViT_B_16_Weights
		:members:
	"""
	#weights = ViT_B_16_Weights.verify(weights)

	return _vision_transformer(
		patch_size=14,
		num_layers=8,
		num_heads=4,
		hidden_dim=96,
		mlp_dim=2048,
#        representation_size=64,
		weights=None,
		progress=progress,
		**kwargs,
	)

# --------------------------------------------------------------------------------------------------------

def vit_mb(*, weights: Optional = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
	"""
	Constructs a vit_b_16 architecture from
	`An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

	Args:
		weights (:class:`~torchvision.models.ViT_B_16_Weights`, optional): The pretrained
			weights to use. See :class:`~torchvision.models.ViT_B_16_Weights`
			below for more details and possible values. By default, no pre-trained weights are used.
		progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
		**kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
			base class. Please refer to the `source code
			<https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
			for more details about this class.

	.. autoclass:: torchvision.models.ViT_B_16_Weights
		:members:
	"""
	#weights = ViT_B_16_Weights.verify(weights)

	return _vision_transformer(
		patch_size=28,
		num_layers=1,
		num_heads=4, 
		hidden_dim=32, # must be an integer multiple of num_heads
		mlp_dim=64,
		representation_size=32,
		weights=None,
		progress=progress,
		**kwargs,
	)