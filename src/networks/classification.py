import torch
from norse.torch import LIFCell, LIFParameters, LILinearCell
from abc import ABC

class AbstractClassificationSNN(ABC, torch.nn.Module):

    def __init__(self, n_input, n_output, in_features, out_features):
        super(AbstractClassificationSNN, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.in_features = in_features
        self.out_feature = out_features

class DoubleConvNet(AbstractClassificationSNN):
    def __init__(self, 
        n_input = 1, 
        n_output = 10,
        in_features=32, 
        out_features=64,
        alpha=100):

        super(DoubleConvNet, self).__init__(n_input, n_output, in_features, out_features)



# class ConvNet(torch.nn.Module):
#     def __init__(self, num_channels=1, feature_size=28, method="super", alpha=100, n_classes = 10):
#         super(ConvNet, self).__init__()

#         self.features = int(((feature_size - 4) / 2 - 4) / 2)
#         self.n_classes = n_classes
#         self.conv1 = torch.nn.Conv2d(num_channels, 16, 5, 1)
#         self.conv2 = torch.nn.Conv2d(16, 32, 5, 1)
#         self.flatten = torch.nn.Flatten()
#         self.fc1 = torch.nn.Linear(self.features * self.features * 32, 500)
#         self.lif0 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
#         self.lif1 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
#         self.lif2 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
#         self.out = LILinearCell(500, self.n_classes)

#     def forward(self, x):
#         seq_length = x.shape[0]
#         batch_size = x.shape[1]
#         s0 = s1 = s2 = so = None

#         voltages = torch.zeros(
#             seq_length, batch_size, self.n_classes, device=x.device, dtype=x.dtype
#         )

#         for ts in range(seq_length):
#             z = self.conv1(x[ts, :])
#             z = torch.nn.functional.max_pool2d(z, 2, 2)
#             z, s0 = self.lif0(z, s0)
            
#             z = self.n_classes * self.conv2(z)
#             z = torch.nn.functional.max_pool2d(z, 2, 2)
#             z, s1 = self.lif1(z, s1)

#             z = self.flatten(z)
#             z = self.fc1(z)
#             z, s2 = self.lif2(z, s2)
#             v, so = self.out(torch.nn.functional.relu(z), so)
#             voltages[ts, :, :] = v
#         return voltages