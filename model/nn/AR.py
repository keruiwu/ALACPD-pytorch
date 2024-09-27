import torch

class AR(torch.nn.Module):
    def __init__(self, input_size, device=None):
        super(AR, self).__init__()
        
        self.linear = torch.nn.Linear(input_size, input_size, bias=True)
        self.device = device

    def forward(self, x):
        """
        :param x: torch.Tensor, shape [batch_size, num_features, seq_len].
        :return: torch.Tensor, shape [batch_size, num_features, seq_len].
        """
        if self.device is not None and not x.device == self.device:
            x = x.to(self.device)
        return self.linear(x)
    

if __name__ == '__main__':
    # test AR
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(15,1,10).to(device)
    ar = AR(x.shape[-1], device)
    
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(ar.parameters(), lr=0.001)

    for i in range(10):
        optimizer.zero_grad()
        output = ar(x)
        loss = loss_fn(output, x)
        loss.backward()
        optimizer.step()
        print(f'loss: {loss.item()}')