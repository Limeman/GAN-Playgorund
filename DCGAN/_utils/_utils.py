import torch.nn as nn

def weights_init(model):
    """
        Intializes model weights as per the specification of the DCGAN paper
        
        model: torch.nn
            A pytorch model whose weights are to be initialized
    """
    class_name = model.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)