import torch
import torch.optim as optim
import torch.nn as nn


from .AR import AR
from .ASC_LSTM import ASC_LSTM


class TAEnet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, res, alpha=0.5, device=None):
        """
        :param input_size: size of input data(num_channels)
        :type input_size: int
        :param hidden_size: size of hidden layer
        :type hidden_size: int
        :param seq_len: length of sequence
        :type seq_len: int
        :param res: residual steps
        :type res: int
        :param alpha: weight of residual connection
        :type alpha: float
        :param device: device to run the model
        :type device: torch.device
        """
        super(TAEnet, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.res = res
        self.alpha = alpha

        self.rnn = ASC_LSTM(input_size, hidden_size, seq_len, res, alpha, device).to(device)
        self.ar = AR(seq_len, device).to(device)

    def forward(self, x):
        """
        :param x: input data with shape (batch_size, num_channels, seq_len)
        :type x: torch.Tensor
        """
        ar = self.ar(x)
        rnn = self.rnn(x)

        return ar + rnn
    
    def fit(self, x, iter=1000, lr=0.001, verbose=False):
        """
        :param x: input data with shape (batch_size, num_channels, seq_len)
        :type x: torch.Tensor
        """
        loss = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # progress = tqdm(range(iter))
        for epoch in range(iter):
            optimizer.zero_grad()
            output = self(x)
            loss_val = loss(output, x)
            loss_val.backward()
            optimizer.step()


        