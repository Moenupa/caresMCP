"""Compact U-Net matching the released Stenoz coronary angiography checkpoints."""

from torch import Tensor, cat, nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Return the release's double-convolution batch-normalized block."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Four-stage U-Net with skip connections, as released by uzbtrust/stenoz."""

    def __init__(self, base: int = 32) -> None:
        super().__init__()
        b = base
        self.e1 = _conv_block(1, b)
        self.e2 = _conv_block(b, b * 2)
        self.e3 = _conv_block(b * 2, b * 4)
        self.e4 = _conv_block(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        self.bott = _conv_block(b * 8, b * 16)
        self.u4 = nn.ConvTranspose2d(b * 16, b * 8, 2, 2)
        self.d4 = _conv_block(b * 16, b * 8)
        self.u3 = nn.ConvTranspose2d(b * 8, b * 4, 2, 2)
        self.d3 = _conv_block(b * 8, b * 4)
        self.u2 = nn.ConvTranspose2d(b * 4, b * 2, 2, 2)
        self.d2 = _conv_block(b * 4, b * 2)
        self.u1 = nn.ConvTranspose2d(b * 2, b, 2, 2)
        self.d1 = _conv_block(b * 2, b)
        self.out = nn.Conv2d(b, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        bottom = self.bott(self.pool(e4))
        d = self.d4(cat([self.u4(bottom), e4], 1))
        d = self.d3(cat([self.u3(d), e3], 1))
        d = self.d2(cat([self.u2(d), e2], 1))
        d = self.d1(cat([self.u1(d), e1], 1))
        return self.out(d)
