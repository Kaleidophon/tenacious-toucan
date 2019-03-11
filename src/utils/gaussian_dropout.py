import torch


class GaussianDropout(torch.nn.Module):
    def __init__(self, p):
        super(GaussianDropout, self).__init__()
        self.alpha = torch.Tensor([p / (1 - p)])

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(1, alpha)
            epsilon = torch.randn(x.size()) * self.alpha + 1

            # epsilon = Variable(epsilon)
            if x.is_cuda:
                epsilon = epsilon.cuda()

            return x * epsilon
        else:
            return x
