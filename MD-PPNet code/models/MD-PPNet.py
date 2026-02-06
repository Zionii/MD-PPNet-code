import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ScaleSelector(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2 ** i, stride=2 ** i),
                SEBlock(channels)  # 
            ) for i in range(3)
        ])
        self.selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, len(self.branches), 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        weights = self.selector(x)
        branch_outs = [
            F.interpolate(b(x), size=x.size()[2:], mode='bilinear', align_corners=True)
            for b in self.branches
        ]
        return sum(w * out for w, out in zip(weights.split(1, 1), branch_outs))


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super().__init__()
        self.features = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ELU(inplace=True)
            ) for bin in bins
        ])

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class DenseBlock(nn.Module):
    def __init__(self, in_c, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c + i * growth_rate, growth_rate, kernel_size=3, padding=1),
                nn.BatchNorm2d(growth_rate),
                nn.ELU(inplace=True)
            ) for i in range(num_layers)
        ])

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.scale_selector = ScaleSelector(out_c)  
        self.dense = DenseBlock(in_c, growth_rate=out_c // 4, num_layers=3)
        self.conv1x1 = nn.Conv2d(in_c + 3 * (out_c // 4), out_c, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.elu = nn.ELU(inplace=True)

    def forward(self, inputs):
        x = self.dense(inputs)
        x = self.conv1x1(x)
        x = self.bn(x)
        x = self.scale_selector(x) 
        x = self.elu(x)
        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class build_unet(nn.Module):
    def __init__(self, args, in_ch=3, n_classes=3):
        super().__init__()
        self.args = args
        self.in_channels = in_ch
        self.num_classes = n_classes

        """ Encoder """
        self.e1 = encoder_block(self.in_channels, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck with PPM """
        self.ppm = PPM(512, 256, [1, 2, 3, 6])
        self.b = conv_block(512 + 4 * 256, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, n_classes, kernel_size=1, padding=0)

    def forward(self, inputs):
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        p4 = self.ppm(p4)
        b = self.b(p4)

        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)
        return outputs


if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.num_input = 3
            self.num_classes = 3


    from thop import profile

    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet(args, args.num_input, args.num_classes).to(device)

 
    x = torch.randn((2, 3, 512, 512), device=device)


    y = model(x)
    print("Output shape:", y.shape)


    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params / 1e6:.2f}M")

    flops, params = profile(model, inputs=(x,))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")