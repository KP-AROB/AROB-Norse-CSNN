import torch
from norse.torch import LIFCell, LIFParameters, LILinearCell
from abc import ABC


class AbstractClassificationSNN(ABC, torch.nn.Module):

    def __init__(self, in_shape, n_output, in_features, out_features):
        super(AbstractClassificationSNN, self).__init__()
        self.in_shape = in_shape
        self.n_output = n_output
        self.in_features = in_features
        self.out_feature = out_features


class DoubleConvNet(AbstractClassificationSNN):
    def __init__(self,
                 in_shape=(1, 28, 28),
                 n_output=10,
                 in_features=32,
                 out_features=64,
                 alpha=100):

        super(DoubleConvNet, self).__init__(
            in_shape, n_output, in_features, out_features)

        self.conv1 = torch.nn.Conv2d(in_shape[0], in_features, 3, 1)
        self.conv2 = torch.nn.Conv2d(in_features, out_features, 3, 1)
        conv1_output_size = in_shape[1] - 2
        pooled1_output_size = conv1_output_size // 2
        conv2_output_size = pooled1_output_size - 2
        pooled2_output_size = conv2_output_size // 2

        self.fc = torch.nn.Linear(
            pooled2_output_size**2 * out_features, out_features)

        self.out = LILinearCell(out_features, n_output)
        self.lif0 = LIFCell(p=LIFParameters(method="super", alpha=alpha))
        self.lif1 = LIFCell(p=LIFParameters(method="super", alpha=alpha))
        self.lif2 = LIFCell(p=LIFParameters(method="super", alpha=alpha))

    def forward(self, x):
        seq_length = x.shape[0]
        batch_size = x.shape[1]
        s0 = s1 = s2 = so = None

        voltages = torch.zeros(
            seq_length, batch_size, self.n_output, device=x.device, dtype=x.dtype
        )

        for ts in range(seq_length):
            z = self.conv1(x[ts, :])
            z = torch.nn.functional.max_pool2d(z, 2, 2)
            z, s0 = self.lif0(z, s0)

            z = self.n_output * self.conv2(z)
            z = torch.nn.functional.max_pool2d(z, 2, 2)
            z, s1 = self.lif1(z, s1)

            z = z.view(batch_size, -1)
            z = self.fc(z)
            z, s2 = self.lif2(z, s2)
            v, so = self.out(torch.nn.functional.relu(z), so)
            voltages[ts, :, :] = v
        return voltages
