import math
import torch

import torch.nn as nn

from .utils import Builder, filter_kwargs_for_module


# Transformer
class Transformer(nn.Module):
    def __init__(self,
                 timestep: int,
                 dimentions: int,
                 num_features: int,
                 num_layers: int=2,
                 nhead: int=4,
                 dim_feedforward: int=512,
                 dropout: float=0.3,
                 pick_time_index: int=-1,
                 **kwargs):
        super(Transformer, self).__init__()

        __kwargs = filter_kwargs_for_module(TimeSeriesTransformer, **kwargs)

        self.transformer = TimeSeriesTransformer(timestep=timestep,
                                                 dimentions=dimentions,
                                                 d_model=num_features,
                                                 nhead=nhead,
                                                 num_layers=num_layers,
                                                 dim_feedforward=dim_feedforward,
                                                 dropout=dropout,
                                                 **__kwargs)
        
        if (pick_time_index > timestep - 1) or (pick_time_index < -timestep):
            pick_time_index = -1
        self.pick_time_index = pick_time_index
                
    def forward(self, x):
        x = self.transformer(x)
        x = x[:, self.pick_time_index, :]
        return x


# Transformer Builder
class TransformerBuilder(Builder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        transformer_kwargs = filter_kwargs_for_module(Transformer, **kwargs)
        self.backbone = Transformer(**transformer_kwargs)


class TimeSeriesTransformer(nn.Module):
    def __init__(self,
                 timestep: int,
                 dimentions: int,
                 d_model=256,
                 nhead=4,
                 num_layers=2,
                 dim_feedforward=512,
                 dropout=0.3):
        super(TimeSeriesTransformer, self).__init__()

        self.d_model = d_model
        self.input_shape = (timestep, dimentions)

        self.embedding = nn.Linear(dimentions, d_model)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.transpose(0,1))

    def forward(self, x):
        x = x + self.pe[:,:x.size(1),:]
        return self.dropout(x)


