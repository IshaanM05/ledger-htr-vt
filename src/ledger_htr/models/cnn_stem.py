import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride=1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class HTRConvStem(nn.Module):
    """Convolutional feature extractor turning a (B,1,H,W) grayscale line
    crop into a sequence of column features for a Transformer+CTC head.

    Downsamples height far more aggressively than width: a single text line
    carries no vertical sequence information (height should collapse toward
    the text's stroke extent), while width is the CTC time axis and needs to
    stay long enough to hold every character with room for CTC's blank
    separators between repeated characters. With target_height=64 this stem
    collapses height to 1 row (64x reduction) and width to 1/4 (matching
    configs/htrvt_fold0*.yaml's max_width, e.g. 1024 -> 256 timesteps).

    Independently designed for this project rather than ported from any
    specific published stem -- see docs/plan.md's License audit for why
    (the reference HTR-VT repo ships without a LICENSE file)."""

    def __init__(self, out_channels: int = 768):
        super().__init__()
        c1, c2, c3 = out_channels // 4, out_channels // 2, out_channels
        self.stem = ConvBNAct(1, c1, kernel_size=3, stride=(2, 1))
        self.pool = nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1)
        self.stage1 = ResidualBlock(c1, c1, stride=(2, 1))
        self.stage2 = ResidualBlock(c1, c2, stride=2)
        self.stage3 = ResidualBlock(c2, c3, stride=2)
        self.final_pool = nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.final_pool(x)
        return x
