import torch


class GaussianDropout(torch.nn.Module):
    def __init__(self, p):
        super(GaussianDropout, self).__init__()
        self.alpha = torch.cuda.FloatTensor([1 / (1 - p)])

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            if x.is_cuda:
                epsilon = torch.cuda.FloatTensor(*x.size()).normal_() * self.alpha + 1
            else:
                epsilon = torch.randn(x.size()) * self.alpha + 1

            # epsilon = Variable(epsilon)

            return x * epsilon
        else:
            return x

