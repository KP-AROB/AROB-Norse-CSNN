import torch


class CLSModel(torch.nn.Module):
    def __init__(self, encoder, snn, decoder):
        super(CLSModel, self).__init__()
        self.encoder = encoder
        self.snn = snn
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.snn(x)
        log_p_y = self.decoder(x)
        return log_p_y
