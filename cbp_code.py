from torch import optim
from GnT_code import GnT

import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch


class ContinualBackprop(object):

    def __init__(
            self,
            net,
            loss,
            step_size=0.001,
            # loss='mse',
            opt='sgd',
            beta=0.9,
            beta_2=0.9,
            replacement_rate=0.001,
            decay_rate=0.9999,
            device='cpu',
            maturity_threshold=100,
            util_type='contribution',
            init='kaiming',
            accumulate=True,
            momentum=0,
            outgoing_random=False,
            weight_decay=0,
            num_epochs = 2000
    ):
        self.net = net
        self.num_epochs = num_epochs


        if opt == 'sgd':
            self.opt = optim.SGD(self.net.parameters(), lr=step_size, momentum=momentum, weight_decay=weight_decay)


        self.loss_func = loss

        self.previous_features = None

        self.gnt = None
        self.gnt = GnT(
            net=self.net.layers,
            hidden_activation=self.net.act_type,
            opt=self.opt,
            replacement_rate=replacement_rate,
            decay_rate=decay_rate,
            maturity_threshold=maturity_threshold,
            util_type=util_type,
            device=device,
            loss_func=self.loss_func,
            init=init,
            accumulate=accumulate,
        )

    def learn(self, x_batch_ve, y_batch_ve, y_hat_batch_ve, select_prob, reward_curr):
        """
        Learn using one step of gradient-descent and generate-&-test
        :param x: input
        :param target: desired output
        :return: loss
        """

        output, features = self.net.predict(x_batch_ve, y_batch_ve, y_hat_batch_ve)

        loss = self.loss_func(output, select_prob, reward_curr)
        self.previous_features = features

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.opt.zero_grad()
        if type(self.gnt) is GnT:
            self.gnt.gen_and_test(features=self.previous_features)
