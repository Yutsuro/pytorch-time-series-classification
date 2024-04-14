import torch.nn as nn

from .head import BasicClassificationHead
from .utils import Builder, filter_kwargs_for_module


# Bidirectional LSTM
class BiLSTM(nn.Module):
    def __init__(self,
                 dimentions: int,
                 num_features: int,
                 num_layers: int=1,
                 **kwargs):
        super(BiLSTM, self).__init__()

        __kwargs = filter_kwargs_for_module(nn.LSTM, **kwargs)

        self.lstm = nn.LSTM(input_size=dimentions,
                            hidden_size=num_features,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            **__kwargs)
        self.layer_norm = nn.LayerNorm(num_features * 2)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.layer_norm(x)
        x = x[:, -1, :]
        return x


# Bidirectional LSTM Builder
class BiLSTMBuilder(Builder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        bilstm_kwargs = filter_kwargs_for_module(BiLSTM, **kwargs)
        self.backbone = BiLSTM(**bilstm_kwargs)

    def build_head(self):
        return BasicClassificationHead(num_classes=self.num_classes,
                                       num_features=self.num_features * 2,
                                       dropout_rate=self.dropout_rate)

