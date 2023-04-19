from gcn.layers import *
from gcn.metrics import *
import torch 
import torch.optim as optim
import torch.nn.functional as F

"""
We: 
- replaced tf with torch for all Tensorflow functions
- replaced tf.nn.relu with F.relu
- replaced tf.nn.softmax with F.softmax
- replaced tf.nn.l2_loss with torch.norm(var, 2)
- replaced Tensorflow's AdamOptimizer with PyTorch's Adam optimizer.
"""

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with torch.nn.ModuleList(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        self.vars = {name: param for name, param in self.named_parameters()}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self):
        save_path = "tmp/%s.pth" % self.name
        torch.save(self.state_dict(), save_path)
        print("Model saved in file: %s" % save_path)

    def load(self):
        save_path = "tmp/%s.pth" % self.name
        self.load_state_dict(torch.load(save_path))
        print("Model restored from file: %s" % save_path)

class MLP(Model):
    def __init__(self, placeholders, input_dim, learning_rate=0.01, weight_decay=0.0, hidden1=16, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].size()[1]
        self.placeholders = placeholders
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.hidden1 = hidden1

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += self.weight_decay * torch.norm(var, 2)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=self.hidden1,
                                 placeholders=self.placeholders,
                                 act=F.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=self.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return F.softmax(self.outputs, dim=1)

class GCN(torch.nn.Module):
    def __init__(self, placeholders, input_dim, learning_rate=0.01, weight_decay=5e-4, hidden1=16):
        super(GCN, self).__init__()

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].size()[1]
        self.placeholders = placeholders

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.weight_decay = weight_decay
        self.hidden1 = hidden1

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += self.weight_decay * torch.norm(var, 2)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def build(self):

        self.layers = torch.nn.ModuleList()

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.hidden1,
                                            placeholders=self.placeholders,
                                            act=F.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=self.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def forward(self):
        for layer in self.layers:
            self.outputs = layer(self.inputs)
            self.inputs = self.outputs

        return F.softmax(self.outputs, dim=1)