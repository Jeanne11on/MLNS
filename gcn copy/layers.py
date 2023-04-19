from gcn.inits import *

import torch
import torch.nn as nn
import torch.nn.functional as F


_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += torch.rand(noise_shape)
    dropout_mask = random_tensor.floor().bool()
    pre_out = x * dropout_mask.float()
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for torch.matmul (sparse vs dense)."""
    if sparse:
        res = torch.sparse.mm(x, y)
    else:
        res = torch.matmul(x, y)
    return res


class Layer(nn.Module):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        forward(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        super(Layer, self).__init__()
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = nn.ParameterDict()
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def forward(self, inputs):
        with nn.parameter_scope(self.name):
            if self.logging and not self.sparse_inputs:
                # Equivalent PyTorch code for tf.summary.histogram
                # Note: PyTorch's equivalent is more flexible and allows for more customization
                if self.logger is not None:
                    self.logger.add_histogram(f'{self.name}/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                # Equivalent PyTorch code for tf.summary.histogram
                # Note: PyTorch's equivalent is more flexible and allows for more customization
                if self.logger is not None:
                    self.logger.add_histogram(f'{self.name}/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            # Equivalent PyTorch code for tf.summary.histogram
            # Note: PyTorch's equivalent is more flexible and allows for more customization
            if self.logger is not None:
                self.logger.add_histogram(f'{self.name}/vars/{var}', self.vars[var])

class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=F.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with nn.parameter_scope(self.name + '_vars'):
            self.vars['weights'] = nn.Parameter(torch.Tensor(input_dim, output_dim))
            nn.init.xavier_uniform_(self.vars['weights'])
            if self.bias:
                self.vars['bias'] = nn.Parameter(torch.Tensor(output_dim))
                nn.init.zeros_(self.vars['bias'])

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = F.dropout(x, p=1-self.dropout, training=self.training)

        # transform
        output = torch.sparse.mm(x, self.vars['weights'])
        output = output.to_dense()  # convert sparse output to dense

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=F.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with nn.parameter_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = nn.Parameter(torch.Tensor(input_dim, output_dim))
                nn.init.xavier_uniform_(self.vars['weights_' + str(i)])
            if self.bias:
                self.vars['bias'] = nn.Parameter(torch.Tensor(output_dim))
                nn.init.zeros_(self.vars['bias'])

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = F.dropout(x, p=1-self.dropout, training=self.training)

        # convolve
        supports = []
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = torch.mm(x, self.vars['weights_' + str(i)])
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = torch.sparse.mm(self.support[i], pre_sup)
            supports.append(support)
        output = sum(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)