import numpy as np

from googlenet import GoogleNet
from nin import NIN


class ModelSpec(object):
    """
    Define the type of input for a pretrained model.
    """

    def __init__(self, re_initilised_layers=None):
        """
        Class constructor.
        """

        self.re_initilised_layers = re_initilised_layers

    def get_factorised_layers(self,
                              reinit_params,
                              fine_tune_params):
        self.reinit_params = reinit_params
        self.fine_tune_params = fine_tune_params


def std_spec(re_initilised_layers):
    """
    The standard network parameters used post-AlexNet
    Args:
        batch_size: defined batch size for input images
    Returns:
        DataSpec: This is conformed to the spec specified
    """
    return ModelSpec(re_initilised_layers=re_initilised_layers)

MODELS = (
          GoogleNet,
          NIN
          )

MODEL_DATA_SPECS = {
    GoogleNet: std_spec(re_initilised_layers=['logits']),
    NIN: std_spec(re_initilised_layers=['logits'])
}


def get_models():
    """Get a list of the availible models"""
    return MODELS


def load_model_template(model_name):
    """

    Args:
        model_name:

    Returns:
        tf.Model instance
    """
    if model_name == 'GoogleNet':
        return GoogleNet
    if model_name == 'NIN':
        return NIN
    raise ValueError('Model: {} is not supported'.format(model_name))
    pass


def load_model_with_data(model_name, input_data, num_classes, train=True):
    """

    Args:
        model_name:
        input_data:

    Returns:

    """
    if model_name == 'GoogleNet':
        return GoogleNet({'data':input_data}, num_classes=num_classes,
                                              trainable=train,
                                              name=model_name)
    if model_name == 'NIN':
        return NIN({'data':input_data}, num_classes=num_classes,
                                        name=model_name)
    raise ValueError('Model: {} is not supported'.format(model_name))
    pass


def get_data_spec(model_instance=None, model_class=None):
    """

    Args:
        model_instance:
        model_class:

    Returns:

    """
    model_class = model_class or model_instance.__class__
    return MODEL_DATA_SPECS[model_class]
