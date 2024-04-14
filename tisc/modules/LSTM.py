import torch.nn as nn

from .utils import Builder, filter_kwargs_for_module


# Normal LSTM
class LSTM(nn.Module):
    def __init__(self,
                 dimentions: int,
                 num_features: int,
                 num_layers: int=1,
                 **kwargs):
        super(LSTM, self).__init__()

        __kwargs = filter_kwargs_for_module(LSTM, **kwargs)

        self.lstm = nn.LSTM(input_size=dimentions,
                            hidden_size=num_features,
                            num_layers=num_layers,
                            batch_first=True,
                            **__kwargs)
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.layer_norm(x)
        x = x[:, -1, :]
        return x


# Normal LSTM Builder
class LSTMBuilder(Builder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        lstm_kwargs = filter_kwargs_for_module(LSTM, **kwargs)
        self.backbone = LSTM(**lstm_kwargs)


