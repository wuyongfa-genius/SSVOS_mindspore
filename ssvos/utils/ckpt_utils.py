"""Implement some func about checkpoint that may be used"""
import os
from mindspore import nn
from ssvos.models.backbones import VideoTransformerNetwork
from mindspore.train import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net


def get_backbone_from_model(model:Model, key_chain:list) -> nn.Cell:
    """Obtain the backbone from a wrapped mindspore Model using the
    key chain provided.
    
    Args:
        model(Model): A Model instance with wrapped network and loss.
        key_chain(list[str]): the keys in the right order according to
            to which we can get backbone.
    Returns:
        The desired backbone(nn.Cell)."""
    network = model.train_network
    # if network is a WithLossCell
    if getattr(model, '_loss_fn') is None:
        assert hasattr(network, '_net')
        network = getattr(network, '_net')
    for key in key_chain:
        assert hasattr(network, key), f'network has no attr named {key}'
        network = getattr(network, key)
    
    return network


def load_pretrained_ckpt(model, ckpt_path, backbone_key_chain=None):
    """Load checkpoint from a pretrained weight.
    
    Args:
        model(Model | Cell): a mindspore Model instance or Cell instance.
        ckpt_path(str): path to the pretrained weight, ckpt is expected
            to have at least the backbone weight.
        backbone_key_chain(list[str]): if you only want to load backbone
            weight, you can specify the keys in order to get the backbone."""
    assert os.path.exists(ckpt_path)
    if backbone_key_chain is not None:
        model = get_backbone_from_model(model, backbone_key_chain)
    load_checkpoint(ckpt_path, model)


def resume_from_ckpt(model, optimizer, ckpt_path):
    """Resume from a saved checkpoint.
    
    Args:
        model(nn.Cell): expected to be a non-wrapped Cell.
        optimizer(Optimizer): an Optimizer instance.
        ckpt_path(str): path to the saved ckpt, expected to contain model 
            params and optimizer params."""
    assert isinstance(model, nn.Cell), 'make sure that you load params into \
        a non-wrapped Cell.'
    assert os.path.exists(ckpt_path)
    ckpt = load_checkpoint(ckpt_path)
    # load param into model
    load_param_into_net(model, ckpt)
    # load param into optimizer
    load_param_into_net(optimizer, ckpt)
    