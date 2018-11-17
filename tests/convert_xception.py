import mxnet as mx
from gluon2keras import gluon2keras
import numpy as np


def check_error(gluon_output, k_model, input_np, epsilon=1e-4):
    """
    Function for outputs comparison.
    :param gluon_output: gluon model output
    :param k_model: keras model
    :param input_np: input numpy array
    :param epsilon: eps
    :return: difference
    """
    gluon_output =  gluon_output.asnumpy()
    keras_output = k_model.predict(input_np)

    error = np.max(gluon_output - keras_output)
    print('Error:', error)

    assert error < epsilon
    return error


if __name__ == '__main__':
    print('Test xception...')

    # Get a model from gluon cv
    from gluoncv2.model_provider import get_model as glcv2_get_model
    net = glcv2_get_model("xception")

    # Make sure it's hybrid and initialized
    net.hybridize()
    net.collect_params().initialize()

    # Test input
    input_np = np.random.uniform(0, 1, (1, 3, 299, 299))
    gluon_output = net(mx.nd.array(input_np))

    # Keras model
    k_model = gluon2keras(net, [(1, 3, 299, 299)], verbose=True, names='short')
    error = check_error(gluon_output, k_model, input_np)
