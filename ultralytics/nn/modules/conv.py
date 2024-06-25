import math
import torch
import torch.nn as nn

__all__ = (
    "Conv",
    "Concat",
    "Detect",
    "SPPF",
    "C2f",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
        print("auto")
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


class Detect(nn.Module):
    """YOLOv8 Detect head module."""

    def __init__(self, nc=80, ch=()):  # number of classes, channels
        super().__init__()
        self.nc = nc
        self.no = nc + 5  # number of outputs per anchor
        self.m = nn.ModuleList([nn.Conv2d(x, self.no, 1) for x in ch])  # output conv
        self.stride = None  # strides computed during build

    def forward(self, x):
        """Forward pass."""
        z = []
        for i in range(len(x)):
            z.append(self.m[i](x[i]))
        return z


class SPPF(nn.Module):
    """SPPF layer for YOLOv8."""

    def __init__(self, c1, c2, k=5):  # ch_in, ch_out, kernal
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1 * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x, self.m(x), self.m(self.m(x)), self.m(self.m(self.m(x)))], 1))


class C2f(nn.Module):
    """C2f module."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * c_, 1, 1)
        self.cv2 = nn.Conv2d(2 * c_, c2, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.m = nn.ModuleList(nn.Conv2d(c_, c_, 3, 1, autopad(3), groups=g, bias=False) for _ in range(n))
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass."""
        y = list(self.cv1(x).chunk(2, 1))
        y[1] = self.m[0](y[1])
        return self.act(self.bn(self.cv2(torch.cat((y[0], y[1]), 1)))) if self.add else self.act(self.bn(self.cv2(torch.cat((y[0], y[1]), 1))))


# The above modules are specifically used in the provided YOLOv8 architecture.
