import inspect

from torch import nn

from ..head import BasicClassificationHead, IdentityHead


class Builder:
    def __init__(self,
                 num_classes: int,
                 num_features: int,
                 dropout_rate: float=0.2,
                 custom_head: nn.Module=None,
                 **kwargs):
        self.num_classes = num_classes
        self.num_features = num_features
        self.dropout_rate = dropout_rate
        self.backbone = None
        
        if custom_head is None:
            if num_classes == 0:
                self.head = IdentityHead()
            else:
                self.head = self.build_head()
        else:
            if isinstance(custom_head, nn.Module):
                self.head = custom_head
            else:
                raise ValueError("Invalid custom_head. Must be a subclass of torch.nn.Module.")

    def build_head(self):
        return BasicClassificationHead(num_classes=self.num_classes,
                                       num_features=self.num_features,
                                       dropout_rate=self.dropout_rate)
    
    def build(self):
        if self.backbone is None:
            raise ValueError("Backbone not defined.")
        return TiscNetwork(self.backbone, self.head)


class TiscNetwork(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 head: nn.Module,
                 ):
        super(TiscNetwork, self).__init__()

        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


def filter_kwargs_for_module(module_class, **kwargs):
        sig = inspect.signature(module_class.__init__)
        module_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return module_kwargs