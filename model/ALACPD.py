import torch
import numpy as np
from tqdm import tqdm

from .nn.TAEnet import TAEnet


class ALACPD():
    def __init__(self, input_size, hidden_size, window_size, res_li=[1], alpha=0.5, C=1.4, beta=0.6, n_cpd=20, device=None):
        """
        :param input_size: size of input data(num_channels)
        :type input_size: int
        :param hidden_size: size of hidden layer
        :type hidden_size: int
        :param window_size: sequence length of sliding window
        :type window_size: int
        :param res_li: list of residual steps
        :type res_li: list
        :param alpha: weight of residual connection
        :type alpha: float
        :param C: threshold for change point detection, threshold=mean(loss) * C
        :type C: float
        :param beta: threshold for how many models should detect anomaly
        :type beta: float
        :param n_cpd: number of consecutive anomalous points to be detected as change point
        :type n_cpd: int
        :param device: device to run the model
        :type device: torch.device
        """
        assert device is not None, "Please specify 'device'!"
        self.device = device

        # model
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.res_li = res_li
        self.alpha = alpha

        # cpd
        self.C = C
        self.beta = beta
        self.n_cpd = n_cpd
        self.th_li = None
        self.stable_cnt = 0

        self.models = [TAEnet(
            input_size, 
            hidden_size, 
            window_size, 
            res, 
            alpha, 
            device).to(device) for res in res_li
        ]

    def fit(self, x, iter=1000, lr=0.001):
        """
        :param x: input data with shape (batch_size, num_channels, seq_len)
        :type x: torch.Tensor
        """
        for model in self.models:
            model.fit(x, iter, lr)
    
    def _calc_th(self, x, loss_li):
        loss_li = np.array(loss_li)

        if self.th_li is None:
            self.th_li = loss_li * self.C
            self.stable_cnt = x.shape[0] * x.shape[2]
        else:
            self.th_li = (self.stable_cnt * self.th_li + loss_li) / (self.stable_cnt + x.shape[0] * x.shape[2])
            self.stable_cnt += x.shape[0] * x.shape[2]

    def _train(self, x, iter):
        self.th_li = None
        loss_li = []
        for model in self.models:
            model.fit(x, iter)
            loss_li.append(torch.nn.MSELoss()(model(x), x).item())
        self._calc_th(x, loss_li)

        print('th_li:', self.th_li)
        print('loss_li:', loss_li)

    def detect(self, x, train_ratio=0.01, n_init=10, n_train=1, n_reinit=50):
        """
        :param x: input data with shape (num_channels, seq_len)
        :type x: torch.Tensor
        """
        print('x:', x.shape)
        # normalize
        if x.mean() != 0 and x.std() != 0:
            x = (x - x.mean()) / x.std()
        # split data
        train_len = int(x.shape[1] * train_ratio)
        print('train_len:', train_len)
        train_x = x[:, :train_len]
        # batch data
        batch_train_x = None
        for idx in range(train_x.shape[1] - self.window_size):
            if batch_train_x is None:
                batch_train_x = train_x[:, idx:idx + self.window_size].unsqueeze(0)
            else:
                batch_train_x = torch.cat(
                    (batch_train_x, train_x[:, idx:idx + self.window_size].unsqueeze(0)), 
                    dim=0
                )
        print('batch_train_x:', batch_train_x.shape)


        # fit
        self._train(batch_train_x, n_init)
        # detect
        cpd_li = []
        cnt = None
        progress = tqdm(range(train_len, x.shape[1] - self.window_size))

        for idx in progress:
            progress.set_description(f'idx: {idx}; cnt: {cnt.shape[0] if cnt is not None else 0}; num_cpd: {len(cpd_li)}')
            test_x = x[:, idx:idx + self.window_size].unsqueeze(0)
            loss_li = [torch.nn.MSELoss()(model(test_x), test_x).item() for model in self.models]
            # print('loss_li:', loss_li)
            # print('th_li:', self.th_li)
            big_loss_cnt = 0
            for loss_idx, loss in enumerate(loss_li):
                if loss > self.th_li[loss_idx]:
                    big_loss_cnt += 1
            if big_loss_cnt / len(self.models) > self.beta:  # anomalous point detected
                if cnt is None:
                    cnt = test_x
                else:
                    cnt = torch.cat((cnt, test_x), dim=0)
                if cnt.shape[0] >= self.n_cpd:  # change point detected
                    cpd_li.append(idx + self.window_size - self.n_cpd)
                    # plt.plot(x[0].cpu().numpy())
                    # plt.axvline(x=train_len, color='g')
                    # plt.axvline(x=idx - cnt, color='r', linestyle='--')
                    # plt.show()
                    # plt.close()
                    # reinitialize models
                    self.models = [TAEnet(
                        self.input_size, 
                        self.hidden_size, 
                        self.window_size, 
                        res, 
                        self.alpha, 
                        self.device).to(self.device) for res in self.res_li
                    ]
                    self._train(cnt, n_reinit)
                    cnt = None
            else:
                cnt = None
                loss_li = []
                for model in self.models:
                    model.fit(test_x, n_train)
                    loss_li.append(torch.nn.MSELoss()(model(test_x), test_x).item())
                self._calc_th(test_x, loss_li)

        return cpd_li
    


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

    model = ALACPD(input_size, hidden_size, window_size, alpha, res_li, device=device, C=C)

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