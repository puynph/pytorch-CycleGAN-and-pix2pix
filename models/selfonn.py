from collections.abc import Iterable
from itertools import repeat
import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, cat, no_grad
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out
from typing import Optional, List, Tuple, Union


def randomshift(x,shifts,learnable,max_shift,rounded_shifts):
    # Take the shape of the input
    c, _, h, w = x.size()

    # Clamp the center bias in case of too much shift after back-propagation
    if learnable:
        torch.clamp(shifts, min=-max_shift, max=max_shift)

        # Round the biases to the integer values
        if rounded_shifts:
            torch.round(shifts)

    # Normalize the coordinates to [-1, 1] range which is necessary for the grid
    a_r = shifts[:,:1] / (w/2)
    b_r = shifts[:,1:] / (h/2)

    # Create the transformation matrix
    aff_mtx = torch.eye(3).to(x.device)
    aff_mtx = aff_mtx.repeat(c, 1, 1)
    aff_mtx[..., 0, 2:3] += a_r
    aff_mtx[..., 1, 2:3] += b_r

    # Create the new grid
    grid = F.affine_grid(aff_mtx[..., :2, :3], x.size(), align_corners=False)

    # Interpolate the input values
    x = F.grid_sample(x, grid, mode='bilinear', align_corners=False)

    return x

def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)

_scalar_or_tuple_1 = Union[int, Tuple[int]]
_scalar_or_tuple_2 = Union[int, Tuple[int, int]]


def _reverse_repeat_tuple(t, n):
    return tuple(x for x in reversed(t) for _ in range(n))


class _SelfONNNd(Module):
    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'in_channels',
                     'out_channels', 'kernel_size', 'q']

    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    q: int
    stride: Tuple[int, ...]
    padding: Tuple[int, ...]
    dilation: Tuple[int, ...]
    groups: int
    padding_mode: str
    sampling_factor: int
    dropout: float
    weight: Tensor
    bias: Optional[Tensor]
    transposed: bool
    output_padding: Tuple[int, ...]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 groups: int,
                 bias: bool,
                 q: int,
                 padding_mode,
                 mode,
                 dropout: Optional[float],
                 transposed: bool,
                 output_padding: Tuple[int, ...]):
        super(_SelfONNNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if padding[0] == -1:
            # Automatically calculate the needed padding for each dim
            newpadding = []
            for dimension in range(len(padding)):
                newpadding.append(math.ceil(self.kernel_size[dimension] / 2) - 1)
            self.padding = tuple(padding)
        else:
            self.padding = padding
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        self._reversed_padding_repeated_twice = tuple(x for x in reversed(self.padding) for _ in range(2))
        self.dilation = dilation
        self.groups = groups
        self.q = q
        self.padding_mode = padding_mode
        self.transposed = transposed
        self.output_padding = output_padding
        self.dropout = dropout
        valid_modes = ["fast", "low_mem"]
        if mode not in valid_modes:
            raise ValueError("mode must be one of {}".format(valid_modes))
        self.mode = mode
        
        # print("*")
        # print(transposed)
        # print("*")
        if transposed:
            self.weight = Parameter(torch.empty(
                (q*in_channels, out_channels // groups, *kernel_size)))
        else:
            self.weight = Parameter(torch.empty(
                (out_channels, q*in_channels // groups, *kernel_size)))
        
        
        
        
        
        # self.weight = Parameter(Tensor(
        #     out_channels, q * in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.weight, gain=gain)
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        if self.mode == 'fast':
            return self._forward_fast(x)
        elif self.mode == 'low_mem':
            return self._forward_low_mem(x)

    def _forward_fast(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def _forward_low_mem(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def extra_repr(self):
        repr_string = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
                       ', stride={stride}, q={q}')
        if self.padding != 0:
            repr_string += ', padding={padding}'
        if self.dilation != 1:
            repr_string += ', dilation={dilation}'
        if self.groups != 1:
            repr_string += ', groups={groups}'
        
        if self.bias is None:
            repr_string += ', bias=False'
        if self.padding_mode != 'zeros':
            repr_string += ', padding_mode={padding_mode}'
        return repr_string.format(**self.__dict__)

    def __setstate__(self, state):
        super(_SelfONNNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


class SelfONN2d(_SelfONNNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _scalar_or_tuple_2,
                 stride: _scalar_or_tuple_2 = 1,
                 padding: _scalar_or_tuple_2 = 0,
                 dilation: _scalar_or_tuple_2 = 1,
                 groups: int = 1,
                 bias: bool = True,
                 q: int = 1,
                 padding_mode: str = 'zeros',
                 mode: str = 'fast',
                 dropout: Optional[float] = None) -> None:
        # Transform type from Union[int, Tuple[int, int]] to Tuple[int, int]
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)
        super(SelfONN2d, self).__init__(in_channels, out_channels, kernel_size_,
                                      stride_, padding_, dilation_, groups, bias,
                                      q, padding_mode, mode, dropout,False,_pair(0))

    def _forward_fast(self, x: Tensor) -> Tensor:
        x = cat([(x ** (i + 1)) for i in range(self.q)], dim=1)
        if self.dropout:
            x = F.dropout2d(x, self.dropout, self.training, False)
        if self.padding_mode != 'zeros':
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            x = F.conv2d(x,
                         self.weight,
                         bias=self.bias,
                         stride=self.stride,
                         padding=_pair(0),
                         dilation=self.dilation,
                         groups=self.groups)
        else:
            x = F.conv2d(x,
                         self.weight,
                         bias=self.bias,
                         stride=self.stride,
                         padding=self.padding,
                         dilation=self.dilation,
                         groups=self.groups)
        return x

    def _forward_low_mem(self, x: Tensor) -> Tensor:
        orig_x = x
        x = F.conv2d(orig_x,
                     self.weights[:, :self.in_channels, :, :],
                     bias=None,
                     stride=self.stride,
                     padding=self.padding,
                     dilation=self.dilation)
        inchannels_per_group = self.in_channels // self.groups
        for q in range(1, self.q):
            x_to_power_q = orig_x ** (q + 1)
            if self.dropout:
                x_to_power_q = F.dropout2d(x, self.dropout, self.training, False)
            x += F.conv2d(
                x_to_power_q,
                self.weight[:, (q * inchannels_per_group):((q + 1) * inchannels_per_group), :, :],
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
        if self.bias is not None:
            x += self.bias[None, :, None, None]
        return x


class _SelfONNTransposeNd(_SelfONNNd):
  def __init__(self,in_channels, out_channels, kernel_size_,
                                      stride_, padding_, dilation_, groups, bias,
                                      q, padding_mode, mode, dropout,transposed,output_padding,device=None, dtype=None) -> None:
        if padding_mode != 'zeros':
            raise ValueError('Only "zeros" padding mode is supported for {}'.format(self.__class__.__name__))

        super(_SelfONNTransposeNd, self).__init__(
           in_channels, out_channels, kernel_size_,
                                      stride_, padding_, dilation_, groups, bias,
                                      q, padding_mode, mode, dropout,transposed,output_padding)

    # dilation being an optional parameter is for backwards
    # compatibility
  def _output_padding(self, input: Tensor, output_size: Optional[List[int]],
                    stride: List[int], padding: List[int], kernel_size: List[int],
                    dilation: Optional[List[int]] = None) -> List[int]:
    # print("*")
    # print(self.output_padding)
    # print("*")
    if output_size is None:
        # print("True")
        ret = _single(self.output_padding)
        # print(self.output_padding)# converting to list if was not already
        # print("True")
    else:
        k = input.dim() - 2
        if len(output_size) == k + 2:
            output_size = output_size[2:]
        if len(output_size) != k:
            raise ValueError(
                "output_size must have {} or {} elements (got {})"
                .format(k, k + 2, len(output_size)))

        min_sizes = torch.jit.annotate(List[int], [])
        max_sizes = torch.jit.annotate(List[int], [])
        for d in range(k):
            dim_size = ((input.size(d + 2) - 1) * stride[d] -
                        2 * padding[d] +
                        (dilation[d] if dilation is not None else 1) * (kernel_size[d] - 1) + 1)
            min_sizes.append(dim_size)
            max_sizes.append(min_sizes[d] + stride[d] - 1)

        for i in range(len(output_size)):
            size = output_size[i]
            min_size = min_sizes[i]
            max_size = max_sizes[i]
            if size < min_size or size > max_size:
                raise ValueError((
                    "requested an output size of {}, but valid sizes range "
                    "from {} to {} (for an input of {})").format(
                        output_size, min_sizes, max_sizes, input.size()[2:]))

        res = torch.jit.annotate(List[int], [])
        for d in range(k):
            res.append(output_size[d] - min_sizes[d])

        ret = res
    return ret


class SelfONNTransposed2d(_SelfONNTransposeNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _scalar_or_tuple_2,
                 stride: _scalar_or_tuple_2 = 1,
                 padding: _scalar_or_tuple_2 = 0,
                 dilation: _scalar_or_tuple_2 = 1,
                 groups: int = 1,
                 bias: bool = True,
                 q: int = 1,
                 padding_mode: str = 'zeros',
                 mode: str = 'fast',
                 output_padding: _scalar_or_tuple_1 = 0,
                 dropout: Optional[float] = None) -> None:
        # Transform type from Union[int, Tuple[int, int]] to Tuple[int, int]
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)
        super(SelfONNTransposed2d, self).__init__( in_channels, out_channels, kernel_size_,
                                   stride_, padding_, dilation_, groups, bias,
                                   q, padding_mode, mode, dropout,True,output_padding)

    def _forward_fast(self, x: Tensor) -> Tensor:
        x = cat([(x ** (i + 1)) for i in range(self.q)], dim=1)
        if self.dropout:
            x = F.dropout2d(x, self.dropout, self.training, False)
        if self.padding_mode != 'zeros':
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            x = F.conv_transpose2d(x,
                         self.weight,
                         bias=self.bias,
                         stride=self.stride,
                         padding=_pair(0),
                         dilation=self.dilation,
                         groups=self.groups)
        else:
            x = F.conv_transpose2d(x,
                         self.weight,
                         bias=self.bias,
                         stride=self.stride,
                         padding=self.padding,
                         dilation=self.dilation,
                         groups=self.groups)
        return x

    def _forward_low_mem(self, x: Tensor) -> Tensor:
        orig_x = x
        x = F.conv_transpose1d(orig_x,
                     self.weights[:, :self.in_channels, :, :],
                     bias=None,
                     stride=self.stride,
                     padding=self.padding,
                     dilation=self.dilation)
        inchannels_per_group = self.in_channels // self.groups
        for q in range(1, self.q):
            x_to_power_q = orig_x ** (q + 1)
            if self.dropout:
                x_to_power_q = F.dropout2d(x, self.dropout, self.training, False)
            x += F.conv2d(
                x_to_power_q,
                self.weight[:, (q * inchannels_per_group):((q + 1) * inchannels_per_group), :, :],
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
        if self.bias is not None:
            x += self.bias[None, :, None, None]
        return x