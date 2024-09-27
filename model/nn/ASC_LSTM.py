import torch
import torch.nn as nn


class ASC_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, res, alpha=0.5, device=None):
        super(ASC_LSTM, self).__init__()
        
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.res = res
        self.alpha = alpha

        self.encoder = [nn.LSTMCell(input_size, hidden_size) for _ in range(seq_len)]
        self.encoder = nn.Sequential(*self.encoder)
        self.activation_enc = nn.ELU()

        self.decoder = [nn.LSTMCell(hidden_size, input_size) for _ in range(seq_len)]
        self.decoder = nn.Sequential(*self.decoder)
        self.activation_dec = nn.Tanh()
        
    def forward(self, x):
        """
        :param x: input data with shape (batch_size, num_channels, seq_len)
        :type x: torch.Tensor
        """
        if self.device is not None and not x.device == self.device:
            x = x.to(self.device)
        h = torch.zeros((x.shape[0], self.hidden_size))

        enc_li = []
        for idx in range(x.shape[2]):
            t = x[:, :, idx]
            h, c = self.encoder[idx](t, (h, h))
            enc_li.append(h)
        enc_li = torch.stack(enc_li).to(self.device)
        enc_li = self.activation_enc(enc_li)
        # add residual connection between res steps
        for idx in range(self.seq_len):
            if idx % self.res == 0:
                enc_li[idx] = self.alpha * enc_li[idx] + (1 - self.alpha) * enc_li[idx - self.res]
        
        dec_li = torch.zeros((self.seq_len, x.shape[0], self.input_size))
        for idx in range(enc_li.shape[0]):
            t = enc_li[enc_li.shape[0] - idx - 1]
            h = torch.zeros((x.shape[0], self.input_size))
            c = torch.zeros((x.shape[0], self.input_size))
            h, c = self.decoder[idx](t, (h, c))
            dec_li[dec_li.shape[0] - idx - 1] = h if idx % self.res == 0 else h + dec_li[dec_li.shape[0] - idx]
        dec_li = self.activation_dec(dec_li).to(self.device)

        return dec_li.permute(1, 2, 0)


if __name__ == "__main__":
    # Test ASC_LSTM
    input_size = 3
    hidden_size = 1
    seq_len = 128
    res = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASC_LSTM(input_size, hidden_size, seq_len, res, device=device)
    x = torch.randn(32, input_size, seq_len)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for i in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, x)
        loss.backward()
        optimizer.step()
        print(f"loss: {loss.item()}")