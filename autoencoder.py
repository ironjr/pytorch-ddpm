import torch
import torch.nn as nn
import torch.nn.functional as F



def make_model(args, parent=False):
    return AutoEncoder(20)


class Classification(nn.Module):
    def __init__(self, num_classes):
        super(Classification, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 512, 1, 1, 0, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(512, num_classes, 1, 1, 0, bias=True),
        )
    def forward(self, x):
        out = self.conv(x)
        return out.softmax(dim=1)


class Encoder(nn.Module):
    def __init__(self,):
        super(Encoder, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
            )
            
    def forward(self, x):
        out = self.conv_stack(x)
        return out # + torch.randn_like(out)

class Decoder(nn.Module):
    def __init__(self,):
        super(Decoder, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0)
        )
    def forward(self, x):
        return self.conv_stack(x)

class AutoEncoder(nn.Module):
    def __init__(self, num_classes=20):
        super(AutoEncoder, self).__init__()
        self.num_classes = num_classes
        self.classification = Classification(self.num_classes)
        self.encoder = nn.ModuleList()
        for i in range(self.num_classes):
            self.encoder.append(Encoder())
        self.decoder = Decoder()
        self.L1 = nn.L1Loss()

    def top1(self, t):
        values, index = t.topk(k=1, dim=1)
        values, index = map(lambda x: x.squeeze(dim=1), (values, index))
        return values, index

    def load_balancing_loss(self, routing):
        _, index = self.top1(routing)
        mask = F.one_hot(index, routing.shape[1]).float()
        mask = mask.reshape(mask.shape[0], -1, mask.shape[-1])
        density = mask.mean(dim=1)
        routing = routing.reshape(routing.shape[0], routing.shape[1], -1)
        density_proxy = routing.mean(dim=-1)
        balancing_loss = (density_proxy * density).mean() * float(routing.shape[1] ** 2)
        return balancing_loss

    def encode(self, x, get_classes=False):
        classes = self.classification(x)
        classes_top = torch.max(classes, dim=1)[1].unsqueeze(1).float()
        # init
        emb = self.encoder[0](x)
        for i in range(1, self.num_classes):
            emb = torch.where(classes_top == i, self.encoder[i](x), emb)
        if get_classes:
            return emb, classes
        return emb

    def decode(self, emb):
        return self.decoder(emb)

    def forward(self, x, loss_weight=0.01):
        """Training code for the autoencoder.

        The loss being used is `1 * (L1) + 0.1 * (Load Balancing Loss)`.
        L1 loss is calculated in the outside of the model.
        """
        emb, classes = self.encode(x, True)
        emb = emb + torch.randn_like(emb)
        out = self.decode(emb)
        
        load_balancing_loss = self.load_balancing_loss(classes)
        return out, load_balancing_loss * loss_weight
