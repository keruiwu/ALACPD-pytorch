import torch
from model.ALACPD import ALACPD


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # test
    input_size = 1
    hidden_size = 20
    window_size = 10
    alpha = 0.5
    C = 2.5
    res_li = [3, 5, 7]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ALACPD(input_size, hidden_size, window_size, res_li, device=device, C=C)

    x = torch.randn(1, input_size, 200)
    temp = torch.randn(1, input_size, 300) + 3
    x = torch.cat((x, temp), dim=2)
    temp = torch.randn(1, input_size, 100)
    x = torch.cat((x, temp), dim=2)

    x = x.squeeze(0)
    cpd_li = model.detect(x, train_ratio=0.05)
    print(cpd_li)
    
    plt.plot(x[0].cpu().numpy())
    for cpd in cpd_li:
        plt.axvline(x=cpd, color='r', linestyle='--')
    plt.show()
    plt.close()