"""06 — ELITE 공식 MGPM network를 원본 연산 그대로 읽고 처음부터 학습한다.

읽는 순서: Mesh2Gaussian.forward -> ExprEncoder -> SongUNet -> UNetBlock
         -> UNetWBConcat -> weight_norm_wrapper/glorot.
GUIDE.md의 '06: Network를 입력에서 gradient까지 읽기'가 이 코드의 상세 해설이다.
파일 위에서부터 읽으면 호환성 helper가 먼저 나온다. 먼저 아래의 forward부터 읽자.

이 lesson에서 답할 질문:
1. 같은 canonical RGB/XYZ인데 프레임마다 Gaussian Map이 어떻게 달라지는가?
2. [B,128] driving이 [B,C,H,W]의 UV 위치별 변화를 어떻게 조건화하는가?
3. Backbone의 16채널 feature와 최종 13+3채널 Gaussian parameter는 어떻게 다른가?
4. Network가 예측하는 residual과 FLAME 자체의 표정/pose 변화는 어디서 합쳐지는가?
5. 고정 decoding/rendering을 지나 loss가 어떤 파라미터까지 돌아가는가?

기본 실행은 초기 forward를 관찰한다. .eval()은 dropout 모드를 바꾸며 학습을 수행하지 않는다.
전체 학습 loop와 optimizer는 08, Gaussian parameter의 물리적 해석은 05에서 읽는다.
공식 learned module의 본문, module 이름, 초기화와 weight-norm 형식을 보존했다.
이번 파일에서는 import 경로를 한 파일로 모으고 forward의 입출력만 lesson에 연결한다.
Fixed Gaussian decoding은 05의 lesson 구현에서 별도로 읽는다. 이 파일은 pretrained weight를 읽지 않는다.

CUDA_VISIBLE_DEVICES=2 python 06_mesh2gaussian_network.py
python 06_mesh2gaussian_network.py --check-parity --device cpu
--check-parity는 별도 원본 source와 초기 state, eval/train forward 및 gradient를 비교한다.
"""

# Ported from kaist-ami/ELITE. Copyright (c) Meta Platforms, Inc. and affiliates.
# Weight-normalized layers, ExprEncoder and UNetWBConcat: Apache-2.0.
# SongUNet/EDM portion: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
# EDM portion: CC BY-NC-SA 4.0. See ELITE_NETWORK_LICENSES.txt.
# Modifications: imports consolidated; Korean commentary; lesson I/O and parity check.

import argparse
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch as th
from torch import nn
import torch.nn.functional as F
import torch.nn.functional as thf
from torch.nn.functional import silu
from torch.nn.modules.utils import _pair
from torch.nn.utils.weight_norm import remove_weight_norm, WeightNorm
from roma import rotvec_to_unitquat

spec = importlib.util.spec_from_file_location("elite_03_entry", Path(__file__).with_name("03_inspect_tracking.py"))
L03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L03)

UPSTREAM_REVISION = "58a7a71dc3e922f589f87882de46154594da0bc3"
MODEL_FORMAT = "elite_official_mgpm_scratch_v2"
UPSTREAM_SHA256 = {'src/models/mesh_unet.py': '18358094423585c3c4ad951cad5367e8cbcc3758fe43d77991153153755dbab9', 'src/nn/layers.py': 'bf314fe16745b2caf517891ee18781eeaaf1e57e0a3a9b3fe815e5fe2d4f25a5', 'src/nn/unet_gs.py': '5878d1a0ad486d69917b0d87eff804082026ffe5d4e8681f9a6b1cb88ced08ed', 'src/nn/unet.py': '7962566abf9781ab7250dc6f324d04cf5d6d9b2f1d5396a2b8a00ab379f4896d', 'configs/3d_prior.yaml': '8c7c917c567bf9adf3c8dcfddad70db5f33bed7138a78363c5031dfb03c7ba33', 'LICENSE': 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30'}
REFERENCE_NODES = {'src/nn/layers.py': ['check_args_shadowing', 'TensorMappingHook', 'weight_norm_wrapper', 'is_weight_norm_wrapped', 'Conv2dUB', 'ConvTranspose2dUB', 'LinearWN', 'Conv2dWNUB', 'ConvTranspose2dWNUB', 'glorot', 'make_linear'], 'src/nn/unet_gs.py': ['weight_init', 'Linear', 'Conv2d', 'GroupNorm', 'AttentionOp', 'PositionalEmbedding', 'FourierEmbedding', 'UNetBlock', 'SongUNet'], 'src/nn/unet.py': ['UNetWBConcat'], 'src/models/mesh_unet.py': ['ExprEncoder']}
DEFAULT_REFERENCE = L03.ROOT / ".cache" / "elite_network_reference" / UPSTREAM_REVISION


# 원본: src/nn/layers.py
# 이곳부터 glorot까지는 encoder/head가 사용하는 layer 및 checkpoint 호환 helper다.
# 처음 읽을 때는 Mesh2Gaussian.forward로 내려가 전체 경로를 본 뒤 돌아온다.
def check_args_shadowing(name, method: object, arg_names) -> None:
    spec = inspect.getfullargspec(method)
    init_args = {*spec.args, *spec.kwonlyargs}
    for arg_name in arg_names:
        if arg_name in init_args:
            raise TypeError(
                f"{name} attempted to shadow a wrapped argument: {arg_name}"
            )

class TensorMappingHook(object):
    def __init__(
        self,
        name_mapping: List[Tuple[str, str]],
        expected_shape: Optional[Dict[str, List[int]]] = None,
    ) -> None:
        """This hook is expected to be used with "_register_load_state_dict_pre_hook" to
        modify names and tensor shapes in the loaded state dictionary.

        Args:
            name_mapping: list of string tuples
            A list of tuples containing expected names from the state dict and names expected
            by the module.

            expected_shape: dict
            A mapping from parameter names to expected tensor shapes.
        """
        self.name_mapping = name_mapping
        self.expected_shape = expected_shape if expected_shape is not None else {}

    def __call__(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        for old_name, new_name in self.name_mapping:
            if prefix + old_name in state_dict:
                tensor = state_dict.pop(prefix + old_name)
                if new_name in self.expected_shape:
                    tensor = tensor.view(*self.expected_shape[new_name])
                state_dict[prefix + new_name] = tensor

# 핵심: g는 output별이지만 v의 norm은 전체 weight로 계산한다. 일반 weight_norm(dim=0)과 다르다.
# 실제 weight W = broadcast(g) * v / ||v||_2. g와 v가 optimizer에 등록되는 Parameter다.
# forward 직전 hook이 W를 계산하므로 W 자체와 weight_g/weight_v를 구분해서 읽는다.
# fuse는 g/v를 실제 W 하나로 합치고, unfuse는 이를 다시 g/v로 분리한다.
# 원본 glorot는 fuse -> W 초기화 -> unfuse 순서로 초기화한다.
def weight_norm_wrapper(
    cls: Type[th.nn.Module],
    new_cls_name: str,
    name: str = "weight",
    g_dim: int = 0,
    v_dim: Optional[int] = 0,
):
    """Wraps a torch.nn.Module class to support weight normalization. The wrapped class
    is compatible with the fuse/unfuse syntax and is able to load state dict from previous
    implementations.

    Args:
        cls: Type[th.nn.Module]
        Class to apply the wrapper to.

        new_cls_name: str
        Name of the new class created by the wrapper. This should be the name
        of whatever variable you assign the result of this function to. Ex:
        ``SomeLayerWN = weight_norm_wrapper(SomeLayer, "SomeLayerWN", ...)``

        name: str
        Name of the parameter to apply weight normalization to.

        g_dim: int
        Learnable dimension of the magnitude tensor. Set to None or -1 for single scalar magnitude.
        Default values for Linear and Conv2d layers are 0s and for ConvTranspose2d layers are 1s.

        v_dim: int
        Of which dimension of the direction tensor is calutated independently for the norm. Set to
        None or -1 for calculating norm over the entire direction tensor (weight tensor). Default
        values for most of the WN layers are None to preserve the existing behavior.
    """

    class Wrap(cls):
        def __init__(
            self, *args: Any, name=name, g_dim=g_dim, v_dim=v_dim, **kwargs: Any
        ):
            # Check if the extra arguments are overwriting arguments for the wrapped class
            check_args_shadowing(
                "weight_norm_wrapper", super().__init__, ["name", "g_dim", "v_dim"]
            )
            super().__init__(*args, **kwargs)

            # Sanitize v_dim since we are hacking the built-in utility to support
            # a non-standard WeightNorm implementation.
            if v_dim is None:
                v_dim = -1
            self.weight_norm_args = {"name": name, "g_dim": g_dim, "v_dim": v_dim}
            self.is_fused = True
            self.unfuse()

            # For backward compatibility.
            self._register_load_state_dict_pre_hook(
                TensorMappingHook(
                    [(name, name + "_v"), ("g", name + "_g")],
                    {name + "_g": getattr(self, name + "_g").shape},
                )
            )

        def fuse(self):
            if self.is_fused:
                return
            # Check if the module is frozen.
            param_name = self.weight_norm_args["name"] + "_g"
            if hasattr(self, param_name) and param_name not in self._parameters:
                raise ValueError("Trying to fuse frozen module.")
            remove_weight_norm(self, self.weight_norm_args["name"])
            self.is_fused = True

        def unfuse(self):
            if not self.is_fused:
                return
            # Check if the module is frozen.
            param_name = self.weight_norm_args["name"]
            if hasattr(self, param_name) and param_name not in self._parameters:
                raise ValueError("Trying to unfuse frozen module.")
            wn = WeightNorm.apply(
                self, self.weight_norm_args["name"], self.weight_norm_args["g_dim"]
            )
            # Overwrite the dim property to support mismatched norm calculate for v and g tensor.
            if wn.dim != self.weight_norm_args["v_dim"]:
                wn.dim = self.weight_norm_args["v_dim"]
                # Adjust the norm values.
                weight = getattr(self, self.weight_norm_args["name"] + "_v")
                norm = getattr(self, self.weight_norm_args["name"] + "_g")
                norm.data[:] = th.norm_except_dim(weight, 2, wn.dim)
            self.is_fused = False

        def __deepcopy__(self, memo):
            # Delete derived tensor to avoid deepcopy error.
            if not self.is_fused:
                delattr(self, self.weight_norm_args["name"])

            # Deepcopy.
            cls = self.__class__
            result = cls.__new__(cls)
            memo[id(self)] = result
            for k, v in self.__dict__.items():
                setattr(result, k, copy.deepcopy(v, memo))

            if not self.is_fused:
                setattr(result, self.weight_norm_args["name"], None)
                setattr(self, self.weight_norm_args["name"], None)
            return result

    # Allows for pickling of the wrapper: https://bugs.python.org/issue13520
    Wrap.__qualname__ = new_cls_name

    return Wrap

def is_weight_norm_wrapped(module) -> bool:
    for hook in module._forward_pre_hooks.values():
        if isinstance(hook, WeightNorm):
            return True
    return False

# Untied bias: bias가 [C,H,W]다. UV 위치마다 독립적인 학습 파라미터를 갖는다.
# 일반 Conv2d는 bias[C]를 모든 위치에 공유하지만 여기서는 output[:,c,h,w]에 bias[c,h,w]를 더한다.
# convolution kernel은 여전히 공간에 공유된다. 위치별 bias는 canonical UV의 고정 대응을 활용하지만
# 공간 평행이동에 대한 일반 convolution의 equivariance를 깨며 해상도별 checkpoint shape를 만든다.
class Conv2dUB(th.nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        *args,
        bias: bool = True,
        **kwargs,
    ) -> None:
        """Conv2d with untied bias."""
        super().__init__(in_channels, out_channels, *args, bias=False, **kwargs)
        self.bias = (
            th.nn.Parameter(th.zeros(out_channels, height, width)) if bias else None
        )

    # TODO: remove this method once upgraded to pytorch 1.8
    def _conv_forward(
        self, input: th.Tensor, weight: th.Tensor, bias: Optional[th.Tensor]
    ):
        # Copied from pt1.8 source code.
        if self.padding_mode != "zeros":
            input = thf.pad(
                input, self._reversed_padding_repeated_twice, mode=self.padding_mode
            )
            return thf.conv2d(
                input, weight, bias, self.stride, _pair(0), self.dilation, self.groups
            )
        return thf.conv2d(
            input,
            weight,
            bias,
            self.stride,
            #  typing.Tuple[int, ...]]` but got `Union[str, typing.Tuple[int, ...]]`.
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, input: th.Tensor) -> th.Tensor:
        output = self._conv_forward(input, self.weight, None)
        bias = self.bias
        if bias is not None:
            # Assertion for jit script.
            assert bias is not None
            # [C,H,W] -> [1,C,H,W]: batch 축에만 broadcast한다.
            output = output + bias[None]
        return output

# 공식 전치 convolution과 PyTorch 버전 호환용 output padding 계산을 함께 보존한다.
class ConvTranspose2dUB(th.nn.ConvTranspose2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        *args,
        bias: bool = True,
        **kwargs,
    ) -> None:
        """ConvTranspose2d with untied bias."""
        super().__init__(in_channels, out_channels, *args, bias=False, **kwargs)

        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose2dUB"
            )

        self.bias = (
            th.nn.Parameter(th.zeros(out_channels, height, width)) if bias else None
        )

    def forward(
        self, input: th.Tensor, output_size: Optional[List[int]] = None
    ) -> th.Tensor:
        # TODO(T111390117): Fix Conv member annotations.
        output_padding = self._output_padding(
            input=input,
            output_size=output_size,
            # `Tuple[int, ...]`.
            stride=self.stride,
            # `Union[str, typing.Tuple[int, ...]]`.
            padding=self.padding,
            # `Tuple[int, ...]`.
            kernel_size=self.kernel_size,
            # This is now required as of D35874490
            num_spatial_dims=input.dim() - 2,
            # `Tuple[int, ...]`.
            dilation=self.dilation,
        )

        output = thf.conv_transpose2d(
            input,
            self.weight,
            None,
            self.stride,
            #  typing.Tuple[int, ...]]` but got `Union[str, typing.Tuple[int, ...]]`.
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
        bias = self.bias
        if bias is not None:
            # Assertion for jit script.
            assert bias is not None
            output = output + bias[None]
        return output

    # NOTE: This function (on super _ConvTransposeNd) was updated in D35874490 with non-optional
    # param num_spatial_dims added. Since we need both old/new pytorch versions to work (until those
    # changes reach DGX), we're simply copying the updated code here until then.
    # TODO remove this function once updated torch code is released to DGX
    def _output_padding(
        self,
        input: th.Tensor,
        output_size: Optional[List[int]],
        stride: List[int],
        padding: List[int],
        kernel_size: List[int],
        num_spatial_dims: int,
        dilation: Optional[List[int]] = None,
    ) -> List[int]:
        if output_size is None:
            # converting to list if was not already
            ret = th.nn.modules.utils._single(self.output_padding)
        else:
            has_batch_dim = input.dim() == num_spatial_dims + 2
            num_non_spatial_dims = 2 if has_batch_dim else 1
            if len(output_size) == num_non_spatial_dims + num_spatial_dims:
                output_size = output_size[num_non_spatial_dims:]
            if len(output_size) != num_spatial_dims:
                raise ValueError(
                    "ConvTranspose{}D: for {}D input, output_size must have {} or {} elements (got {})".format(
                        num_spatial_dims,
                        input.dim(),
                        num_spatial_dims,
                        num_non_spatial_dims + num_spatial_dims,
                        len(output_size),
                    )
                )

            min_sizes = th.jit.annotate(List[int], [])
            max_sizes = th.jit.annotate(List[int], [])
            for d in range(num_spatial_dims):
                dim_size = (
                    (input.size(d + num_non_spatial_dims) - 1) * stride[d]
                    - 2 * padding[d]
                    + (dilation[d] if dilation is not None else 1)
                    * (kernel_size[d] - 1)
                    + 1
                )
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError(
                        (
                            "requested an output size of {}, but valid sizes range "
                            "from {} to {} (for an input of {})"
                        ).format(output_size, min_sizes, max_sizes, input.size()[2:])
                    )

            res = th.jit.annotate(List[int], [])
            for d in range(num_spatial_dims):
                res.append(output_size[d] - min_sizes[d])

            ret = res
        return ret

LinearWN = weight_norm_wrapper(th.nn.Linear, "LinearWN", g_dim=0, v_dim=None)

Conv2dWNUB = weight_norm_wrapper(Conv2dUB, "Conv2dWNUB", g_dim=0, v_dim=None)

ConvTranspose2dWNUB = weight_norm_wrapper(
    ConvTranspose2dUB, "ConvTranspose2dWNUB", g_dim=1, v_dim=None
)

# 공식 초기화: 전치 convolution의 2x2 weight를 공유하도록 초기화한다. 임의 gain을 추가하지 않는다.
# 2x2 값의 동일성은 초기화 순간에만 설정한다. 이후에는 각 weight 원소가 독립적으로 학습된다.
# alpha는 LeakyReLU의 negative slope에 맞춘 gain 계산 인자다. out layer에는 alpha=1을 사용한다.
def glorot(m: th.nn.Module, alpha: float = 1.0) -> None:
    gain = np.sqrt(2.0 / (1.0 + alpha**2))

    if isinstance(m, th.nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, th.nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // 4
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, th.nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] // 8
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, th.nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * np.sqrt(2.0 / (n1 + n2))
    else:
        return

    is_wnw = is_weight_norm_wrapped(m)
    if is_wnw:
        m.fuse()

    m.weight.data.uniform_(-std * np.sqrt(3.0), std * np.sqrt(3.0))
    if m.bias is not None:
        m.bias.data.zero_()

    if isinstance(m, th.nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    if is_wnw:
        m.unfuse()

def make_linear(n_in, n_out, mode, act=None, bias=True):
    if mode == "wn":
        layers = [LinearWN(n_in, n_out, bias=bias)]
        if act is not None:
            layers.append(act)
    return layers

la = SimpleNamespace(LinearWN=LinearWN, Conv2dWNUB=Conv2dWNUB,
                     ConvTranspose2dWNUB=ConvTranspose2dWNUB, glorot=glorot)


# 원본: src/nn/unet_gs.py
def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

class Conv2d(torch.nn.Module):
    def __init__(self,
                 in_channels, out_channels, kernel, bias=True, up=False, down=False,
                 resample_filter=[1, 1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
                 ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels * kernel * kernel, fan_out=out_channels * kernel * kernel)
        self.weight = torch.nn.Parameter(
            weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(
            weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        # [1,1]의 outer product를 정규화하면 2x2의 각 값은 1/4이다.
        # 학습 kernel과 별개인 고정 low-pass filter; buffer이므로 optimizer가 갱신하지 않는다.
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x, N_views_xa=1):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]),
                                                     groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad + f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            # 기본 SongUNet은 이 경로다. up/down의 고정 resampling 뒤에 learned convolution을 한다.
            # down: 채널별 filter + stride=2. up: filter의 transposed convolution으로 공간 크기를 2배.
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]),
                                                         groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels,
                                               stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x

class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x, N_views_xa=1):
        # 각 sample 안에서 채널 group과 공간 축을 정규화한다. Batch size=1에서도 동작한다.
        # weight/bias[C]는 학습되고 running mean/variance는 없다. UB bias[C,H,W]와도 다르다.
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype),
                                           bias=self.bias.to(x.dtype), eps=self.eps)
        return x.to(memory_format=torch.channels_last)

# Q/K softmax와 custom backward를 FP32에서 계산한다. QKV 채널 배치도 원본을 유지한다.
# q/k: [B*heads, C_head, H*W]. w[q,k]는 query UV 위치가 key 위치를 참고하는 비율이다.
# AttentionOp는 attention weight까지만 계산한다. value를 섞는 einsum과 출력 projection은 UNetBlock에 있다.
class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(
            dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2,
                                          input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk

class PositionalEmbedding(torch.nn.Module):
    # 원본에 포함된 선택적 embedding이다. 이번 설정은 channel_mult_noise=0이므로 사용하지 않는다.
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        b, c = x.shape
        x = rearrange(x, 'b c -> (b c)')
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        x = rearrange(x, '(b c) emb_ch -> b (c emb_ch)', b=b)
        return x

class FourierEmbedding(torch.nn.Module):
    # 이것도 이번 실행 경로에서 사용하지 않는다. Driving은 ExprEncoder와 map_layer0/1에서 처리한다.
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        b, c = x.shape
        x = rearrange(x, 'b c -> (b c)')
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        x = rearrange(x, '(b c) emb_ch -> b (c emb_ch)', b=b)
        return x

# driving을 affine projection 후 더한다. adaptive_scale=False의 연속 norm/SiLU도 원본 그대로다.
# UNetBlock 안의 residual addition과 SongUNet의 encoder/decoder skip concatenation은 다른 연산이다.
# residual: 같은 shape의 tensor를 더함. U-Net skip: 공간 크기를 맞춘 tensor를 채널 축으로 연결함.
class UNetBlock(torch.nn.Module):
    def __init__(self,
                 in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
                 num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
                 resample_filter=[1, 1], resample_proj=False, adaptive_scale=True,
                 init=dict(), init_zero=dict(init_weight=0), init_attn=None,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if emb_channels is not None:
            self.affine = Linear(in_features=emb_channels, out_features=out_channels * (2 if adaptive_scale else 1),
                                 **init)
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down,
                            resample_filter=resample_filter, **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down,
                               resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels * 3, kernel=1,
                              **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb=None, N_views_xa=1):
        # 1) 이전 feature를 보관한다. conv0가 공간 크기/채널을 바꾸면 skip도 같은 shape로 바꾼다.
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        if emb is not None:
            # 2) [B,64] -> [B,C_out] -> [B,C_out,1,1]. 같은 driving 값을 모든 UV 위치에 더한다.
            # 위치마다 x가 다르고 이후 convolution/head도 다르게 반응하므로 최종 변화는 공간적으로 다르다.
            params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
            if self.adaptive_scale:
                scale, shift = params.chunk(chunks=2, dim=1)
                x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
            else:
                # 기본 ELITE 경로. 이름 film_emb만 보고 scale+shift 방식이라고 해석하지 말자.
                x = silu(self.norm1(x.add_(params)))

        # 3) norm1/SiLU가 한 번 더 적용된다. 위 conditional norm/SiLU와 합쳐 없애면 다른 network다.
        x = silu(self.norm1(x))

        # 4) dropout은 train에서만 켜진다. Residual 합의 scale은 기본 1/sqrt(2)다.
        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            # 5) 낮은 해상도에서 UV 전체의 관계를 학습한다. 이 lesson은 N_views_xa=1만 사용한다.
            if N_views_xa != 1:
                B, C, H, W = x.shape
                # (B, C, H, W) -> (B/N, N, C, H, W) -> (B/N, N, H, W, C)
                x = x.reshape(B // N_views_xa, N_views_xa, *x.shape[1:]).permute(0, 1, 3, 4, 2)
                # (B/N, N, H, W, C) -> (B/N, N*H, W, C) -> (B/N, C, N*H, W)
                x = x.reshape(B // N_views_xa, N_views_xa * x.shape[2], *x.shape[3:]).permute(0, 3, 1, 2)
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3,
                                                      -1).unbind(2)
            w = AttentionOp.apply(q, k)
            # [query,key] attention으로 각 key의 value를 가중합해 query feature를 만든다.
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
            if N_views_xa != 1:
                # (B/N, C, N*H, W) -> (B/N, N*H, W, C)
                x = x.permute(0, 2, 3, 1)
                # (B/N, N*H, W, C) -> (B/N, N, H, W, C) -> (B/N, N, C, H, W)
                x = x.reshape(B // N_views_xa, N_views_xa, H, W, C).permute(0, 1, 4, 2, 3)
                # (B/N, N, C, H, W) -> # (B, C, H, W)
                x = x.reshape(B, C, H, W)
        return x

# 공식 enc/dec ModuleDict와 모든 skip을 보존한다. 출력은 아직 Gaussian Map이 아닌 16채널 feature다.
# 'encoder/decoder'는 이 클래스 안의 U-Net down/up 경로다. ExprEncoder나 05 fixed decode와 구분한다.
# Diffusion 계열 backbone이지만 이번 호출에는 noise/timestep 입력이나 denoising 반복이 없다.
class SongUNet(nn.Module):
    def __init__(self,
                 img_resolution,  # Image resolution at input/output.
                 in_channels,  # Number of color channels at input.
                 out_channels,  # Number of color channels at output.
                 emb_dim_in=0,  # Input embedding dim.
                 augment_dim=0,  # Augmentation label dimensionality, 0 = no augmentation.

                 model_channels=128,  # Base multiplier for the number of channels.
                 channel_mult=[1, 2, 2, 2],  # Per-resolution multipliers for the number of channels.
                 channel_mult_emb=2,  # Multiplier for the dimensionality of the embedding vector.
                 num_blocks=4,  # Number of residual blocks per resolution.
                 attn_resolutions=[16],  # List of resolutions with self-attention.
                 dropout=0.10,  # Dropout probability of intermediate activations.
                 label_dropout=0,  # Dropout probability of class labels for classifier-free guidance.

                 embedding_type='positional',  # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
                 channel_mult_noise=0,  # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
                 encoder_type='standard',  # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
                 decoder_type='standard',  # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
                 resample_filter=[1, 1],  # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
                 ):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']

        super().__init__()
        self.label_dropout = label_dropout
        self.emb_dim_in = emb_dim_in
        if emb_dim_in > 0:
            emb_channels = model_channels * channel_mult_emb
        else:
            emb_channels = None
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        # 이름 init_zero와 달리 이 설정은 정확한 0이 아니라 1e-5의 작은 weight로 시작한다.
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        # Mapping.
        # self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        # self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        # self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        # self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        if emb_dim_in > 0:
            self.map_layer0 = Linear(in_features=emb_dim_in, out_features=emb_channels, **init)
            self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        if noise_channels > 0:
            self.noise_map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
            self.noise_map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True,
                                                          **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(in_channels=caux, out_channels=caux, kernel=0, down=True,
                                                               resample_filter=resample_filter)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(in_channels=caux, out_channels=cout, kernel=3,
                                                                   down=True, resample_filter=resample_filter,
                                                                   fused_resample=True, **init)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn,
                                                                **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]
        # 기본 512/6단/num_blocks=1에서는 conv+block(2개), 다음 각 down+block(2개): 총 12개 skip.
        # decoder는 해상도당 num_blocks+1=2개의 block으로 이 skip들을 모두 소비한다.

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True,
                                                         **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn,
                                                                **block_kwargs)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res}x{res}_aux_up'] = Conv2d(in_channels=out_channels, out_channels=out_channels,
                                                             kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                # self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, init_weight=0.2, **init)  # init_zero)
                # TODO(youwang): 20250826 added initialize
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3)  # init_zero)
                nn.init.normal_(self.dec[f'{res}x{res}_aux_conv'].weight, mean=0.0, std=1e-3)
                nn.init.constant_(self.dec[f'{res}x{res}_aux_conv'].bias, 0.0)




    def forward(self, x, film_emb=None, N_views_xa=1):

        emb = None

        if film_emb is not None:
            # ExprEncoder의 [B,128] -> [B,64] -> [B,64]; 각각 SiLU. 모든 UNetBlock에 전달한다.
            # if self.emb_dim_in != 1:
            #     film_emb = film_emb.reshape(
            #         film_emb.shape[0], 2, -1).flip(1).reshape(*film_emb.shape)  # swap sin/cos
            film_emb = silu(self.map_layer0(film_emb))
            film_emb = silu(self.map_layer1(film_emb))
            emb = film_emb

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            # 기본 encoder_type='standard'에는 aux_* 분기가 없다. 각 일반 block의 출력을 저장한다.
            if 'aux_down' in name:
                aux = block(aux, N_views_xa)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux, N_views_xa)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux, N_views_xa)) / np.sqrt(2)
            else:
                x = block(x, emb=emb, N_views_xa=N_views_xa) if isinstance(block, UNetBlock) \
                    else block(x, N_views_xa=N_views_xa)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux, N_views_xa)
            elif 'aux_norm' in name:
                tmp = block(x, N_views_xa)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp), N_views_xa)
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    # 예: bottleneck feature 64ch + 같은 16x16의 encoder skip 64ch = 128ch 입력.
                    # pop()은 가장 나중에 저장된 skip부터 꺼낸다. UV 해상도는 같아야 한다.
                    # skip connection is pixel-aligned which is good for
                    # foreground features
                    # but it's not good for gradient flow and background features
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb=emb, N_views_xa=N_views_xa)
        return aux


# 원본: src/nn/unet.py
# 6번 down/up과 원래 입력까지 연결하는 skip. geometry/app head가 각각 독립적인 network다.
# 입력 F_shared[B,16,512,512]; n_init_ftrs=16; head의 F는 채널 수를 나타내는 정수다.
# geometry head와 appearance head는 모양이 비슷해도 weight를 공유하지 않는다.
class UNetWBConcat(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        size: int,
        n_init_ftrs: int = 8,
        overwrite_weightnorm_init: bool = False
    ):
        super().__init__()

        F = n_init_ftrs

        self.size = size

        self.down1 = nn.Sequential(
            la.Conv2dWNUB(in_channels, F, self.size // 2, self.size // 2, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.down2 = nn.Sequential(
            la.Conv2dWNUB(F, 2 * F, self.size // 4, self.size // 4, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.down3 = nn.Sequential(
            la.Conv2dWNUB(2 * F, 4 * F, self.size // 8, self.size // 8, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.down4 = nn.Sequential(
            la.Conv2dWNUB(4 * F, 8 * F, self.size // 16, self.size // 16, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.down5 = nn.Sequential(
            la.Conv2dWNUB(8 * F, 16 * F, self.size // 32, self.size // 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.down6 = nn.Sequential(
            la.Conv2dWNUB(16 * F, 32 * F, self.size // 64, self.size // 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.up0 = nn.Sequential(
            la.ConvTranspose2dWNUB(
                32 * F, 16 * F, self.size // 32, self.size // 32, 4, 2, 1
            ),
            nn.LeakyReLU(0.2),
        )
        self.up1 = nn.Sequential(
            la.ConvTranspose2dWNUB(
                2 * 16 * F, 8 * F, self.size // 16, self.size // 16, 4, 2, 1
            ),
            nn.LeakyReLU(0.2),
        )
        self.up2 = nn.Sequential(
            la.ConvTranspose2dWNUB(
                2 * 8 * F, 4 * F, self.size // 8, self.size // 8, 4, 2, 1
            ),
            nn.LeakyReLU(0.2),
        )
        self.up3 = nn.Sequential(
            la.ConvTranspose2dWNUB(
                2 * 4 * F, 2 * F, self.size // 4, self.size // 4, 4, 2, 1
            ),
            nn.LeakyReLU(0.2),
        )
        self.up4 = nn.Sequential(
            la.ConvTranspose2dWNUB(
                2 * 2 * F, F, self.size // 2, self.size // 2, 4, 2, 1
            ),
            nn.LeakyReLU(0.2),
        )
        self.up5 = nn.Sequential(
            la.ConvTranspose2dWNUB(2 * F, F, self.size, self.size, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        # self.up1 = nn.Sequential(
        #     la.ConvTranspose2dWNUB(
        #         16 * F, 8 * F, self.size // 16, self.size // 16, 4, 2, 1
        #     ),
        #     nn.LeakyReLU(0.2),
        # )
        # self.up2 = nn.Sequential(
        #     la.ConvTranspose2dWNUB(
        #         2 * 8 * F, 4 * F, self.size // 8, self.size // 8, 4, 2, 1
        #     ),
        #     nn.LeakyReLU(0.2),
        # )
        # self.up3 = nn.Sequential(
        #     la.ConvTranspose2dWNUB(
        #         2 * 4 * F, 2 * F, self.size // 4, self.size // 4, 4, 2, 1
        #     ),
        #     nn.LeakyReLU(0.2),
        # )
        # self.up4 = nn.Sequential(
        #     la.ConvTranspose2dWNUB(
        #         2 * 2 * F, F, self.size // 2, self.size // 2, 4, 2, 1
        #     ),
        #     nn.LeakyReLU(0.2),
        # )
        # self.up5 = nn.Sequential(
        #     la.ConvTranspose2dWNUB(2 * F, F, self.size, self.size, 4, 2, 1),
        #     nn.LeakyReLU(0.2),
        # )
        self.out = la.Conv2dWNUB(
            F + in_channels, out_channels, self.size, self.size, kernel_size=1
        )
        self.apply(lambda x: la.glorot(x, 0.2))
        la.glorot(self.out, 1.0)

        if overwrite_weightnorm_init:
            with th.no_grad():
                self.out.weight.mul_(1e-2)
                if hasattr(self.out, 'weight_g'):  # WeightNorm일 때 스케일 파라미터
                    self.out.weight_g.mul_(1e-2)
                if self.out.bias is not None:
                    self.out.bias.zero_()


    def forward(self, x):
        # 다운 경로의 각 feature는 같은 해상도의 업 경로에 다시 연결한다.
        # x1 [B,16,512,512], x2 [B,16,256,256], ..., x7 [B,512,8,8].
        x1 = x
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        # up0(x7) [B,256,16,16] + x6 [B,256,16,16]를 concat -> [B,512,16,16].
        x = th.cat([self.up0(x7), x6], 1)
        x = th.cat([self.up1(x), x5], 1)
        # x = th.cat([self.up1(x6), x5], 1)
        x = th.cat([self.up2(x), x4], 1)
        x = th.cat([self.up3(x), x3], 1)
        x = th.cat([self.up4(x), x2], 1)
        x = self.up5(x)
        # 마지막에 원래 입력까지 연결: [B,16,512,512] 두 개를 concat -> [B,32,512,512].
        x = th.cat([x, x1], dim=1)
        # 1x1 learned conv + untied bias. 여기에는 tanh/sigmoid나 RGB clamp가 없다; 05가 해석한다.
        return self.out(x)


# 원본: src/models/mesh_unet.py
# Expression 100차원과 quaternion pose를 개별 projection한 뒤 128차원 driving embedding을 만든다.
# 'Encoder'는 tracked FLAME 수치를 잠재 vector로 옮긴다는 뜻이다. 영상에서 표정을 추정하지 않는다.
# Shape coefficient는 이 함수의 입력이 아니다. Identity 관련 정보는 canonical RGB/XYZ 입력에 있다.
class ExprEncoder(nn.Module):
    def __init__(
            self,
            n_embs: int,
    ):
        super().__init__()
        self.n_embs = n_embs
        self.proj_expr = nn.Sequential(
            *make_linear(
                100, self.n_embs,
                "wn",
                nn.LeakyReLU(0.2, inplace=True)
            )
        )
        self.proj_rottrans = nn.Sequential(
            *make_linear(
                7, self.n_embs // 4,
                "wn",
                nn.LeakyReLU(0.2, inplace=True)
            )
        )
        self.proj_neck = nn.Sequential(
            *make_linear(
                4, self.n_embs // 4,
                "wn",
                nn.LeakyReLU(0.2, inplace=True)
            )
        )
        self.proj_jaw = nn.Sequential(
            *make_linear(
                4, self.n_embs // 4,
                "wn",
                nn.LeakyReLU(0.2, inplace=True)
            )
        )
        self.proj_eyes = nn.Sequential(
            *make_linear(
                8, self.n_embs // 4,
                "wn",
                nn.LeakyReLU(0.2, inplace=True)
            )
        )
        self.mlp = nn.Sequential(
            la.LinearWN(2 * self.n_embs, 2 * self.n_embs),
            th.nn.LeakyReLU(0.2, inplace=True),
            la.LinearWN(2 * self.n_embs, 2 * self.n_embs),
            th.nn.LeakyReLU(0.2, inplace=True),
            la.LinearWN(2 * self.n_embs, self.n_embs)
        )
        self.apply(lambda m: la.glorot(m, 0.2))

    def forward(self, batch):
        # 입력 shape: expr[B,100], rot/trans/neck/jaw[B,3], eyes[B,6].
        rot = batch['flame_rot']
        trans = batch['flame_trans']
        neck = batch['flame_neck']
        jaw = batch['flame_jaw']
        eyes = batch['flame_eyes']
        expr = batch['flame_expr']

        # 각 항목은 LinearWN + LeakyReLU. Expr 100->128, 나머지 각 branch->32.
        # Roma는 axis-angle[3]을 xyzw quaternion[4]로 바꾸는 고정 미분 가능 함수다.
        # Quaternion은 단일 axis-angle chart의 회전을 다른 표현으로 전달한다; 모든 모호성이 사라지지는 않는다.
        expr_proj = self.proj_expr(expr)
        jaw_proj = self.proj_jaw(rotvec_to_unitquat(jaw))
        leye_quat = rotvec_to_unitquat(eyes[..., :3])
        reye_quat = rotvec_to_unitquat(eyes[..., 3:])
        eyes_proj = self.proj_eyes(th.cat([leye_quat, reye_quat], dim=1))
        rottrans_proj = self.proj_rottrans(th.cat([rotvec_to_unitquat(rot), trans], dim=-1))
        neck_proj = self.proj_neck(rotvec_to_unitquat(neck))

        # [128(expr),32(jaw),32(eyes),32(rot+trans),32(neck)] -> [B,256].
        all_expr_feats = th.cat([expr_proj, jaw_proj, eyes_proj, rottrans_proj, neck_proj], dim=1)
        # 256 -> 256 -> 256 -> 128. 최종 128개의 수에는 사전에 정해진 물리 채널 이름이 없다.
        expr_embs = self.mlp(all_expr_feats)  # [B, d_lat]

        return expr_embs


# 원본 MeshUNetPrimDecoder의 learned module 구성. fixed decoding은 05로 분리한다.
class GaussianMapDecoder(nn.Module):

    def __init__(self, cfg, uv_size, device):
        super().__init__()
        self.cfg = cfg
        self.uv_size = uv_size
        self.device = device
        self.n_splats = uv_size ** 2
        self.gs_type = cfg['decoder']['gs_type']
        self.learn_quat = cfg['decoder']['learn_quat']
        self.learn_stoffset = cfg['decoder']['learn_stoffset']
        try:
            self.learn_opacity = cfg['decoder']['learn_opacity']
        except KeyError:
            self.learn_opacity = False
        self.n_color = 3
        if self.gs_type == '3dgs':
            self.n_gs_param = 3 + 1
        elif self.gs_type == '2dgs':
            self.n_gs_param = 3 + 2
        if self.learn_quat:
            self.n_gs_param += 4
        if self.learn_stoffset:
            self.n_gs_param += 3
        if self.learn_opacity:
            self.n_gs_param += 1
        self.unet_in_channels = 0
        if cfg['decoder']['use_uv_rgb']:
            self.unet_in_channels += 3
        if cfg['decoder']['use_uv_geo']:
            self.unet_in_channels += 3
        self.unet = SongUNet(img_resolution=self.uv_size, in_channels=self.unet_in_channels, out_channels=self.n_color + self.n_gs_param, model_channels=cfg['decoder']['model_channels'], num_blocks=cfg['decoder']['num_blocks'], emb_dim_in=cfg['decoder']['lat_dim'], channel_mult=cfg['decoder']['channel_mult'], channel_mult_noise=0, attn_resolutions=cfg['decoder']['attn_resolutions'])
        self.geo_enhancenet = UNetWBConcat(in_channels=self.n_color + self.n_gs_param, out_channels=self.n_gs_param, n_init_ftrs=self.n_color + self.n_gs_param, size=uv_size)
        self.app_enhancenet = UNetWBConcat(in_channels=self.n_color + self.n_gs_param, out_channels=self.n_color, n_init_ftrs=self.n_color + self.n_gs_param, size=uv_size)

    def forward(self, uv_input, embs):
        # 원본 forward에서 Map을 생성하는 세 연산과 동일하다.
        # [B,6,512,512] + [B,128] -> [B,16,512,512]. 16은 출력 차원이며 feature 의미를 지정하지 않는다.
        pred_unet = self.unet(uv_input, embs)
        # 독립적인 두 head가 shared feature를 각 task의 raw parameter로 바꾼다.
        geo_out = self.geo_enhancenet(pred_unet)
        app_out = self.app_enhancenet(pred_unet)
        # geo[B,13,512,512], app[B,3,512,512]. 아직 valid mask 선택/3D 배치/rendering은 하지 않았다.
        return geo_out, app_out


def official_config(config):
    """Lesson 설정의 이름만 공식 설정으로 연결한다. 공식 2DGS 구성은 그대로다."""
    if config.get("format") != MODEL_FORMAT:
        raise ValueError("이전 재구현 checkpoint/config와 호환되지 않는다. 공식 network용 새 experiment를 사용하세요.")
    n = config["network"]
    decoder = dict(
        use_uv_rgb=True, use_uv_geo=True, lat_dim=n["embedding"], gs_type="2dgs",
        learn_quat=True, learn_stoffset=True, learn_opacity=True,
        learn_input_upsample=False, deeper_input_upsample_layer=False, multigs_perpixel=False,
        unet_type="SongUNet", model_channels=n["model_channels"],
        attn_resolutions=list(n["attention_sizes"]), num_blocks=n["num_blocks"],
        channel_mult=list(n["channel_mult"]), use_uv_dino=False,
    )
    decoder["2dgs_disp_min"], decoder["2dgs_disp_max"] = -.1, .1
    mode = dict(model_type="mesh_unet", uv_size=config["uv_size"])
    return dict(decoder=decoder, training=dict(mode), inference=dict(mode))


class Mesh2Gaussian(nn.Module):
    """공식 encoder/decoder 이름과 learned state_dict를 유지하는 lesson 연결부.

    Encoder와 decoder의 생성 순서도 원본과 같다. Checkpoint는 읽지 않는다.
    아래 property는 08의 gradient 로그를 위한 별칭이며 모듈을 중복 등록하지 않는다.
    """
    def __init__(self, config):
        super().__init__()
        cfg = official_config(config)
        size = config["uv_size"]
        if size < 64 or size % 64:
            raise ValueError("공식 6-level head에는 UV size>=64, 64의 배수가 필요하다.")
        self.encoder = ExprEncoder(n_embs=cfg["decoder"]["lat_dim"])
        self.decoder = GaussianMapDecoder(cfg, size, "cpu")

    @property
    def driving(self):
        return self.encoder

    @property
    def unet(self):
        return self.decoder.unet

    @property
    def geometry_head(self):
        return self.decoder.geo_enhancenet

    @property
    def appearance_head(self):
        return self.decoder.app_enhancenet

    def forward(self, uv_input, parameters):
        # RGB/XYZ는 이미 05.uv_input에서 공식 입력 범위로 변환했다.
        # 여기서 시작하자: 한 identity의 canonical UV는 고정 입력이고 parameters는 프레임별 관측 값이다.
        # lesson 이름 -> 공식 이름을 옮기는 dict다. FLAME forward나 tracking을 다시 수행하지 않는다.
        batch = {
            "flame_expr": parameters["expr"], "flame_rot": parameters["rotation"],
            "flame_trans": parameters["translation"], "flame_neck": parameters["neck_pose"],
            "flame_jaw": parameters["jaw_pose"], "flame_eyes": parameters["eyes_pose"],
        }
        # ① 무엇을 어떻게 움직일지: tracked driving -> learned embedding [B,128].
        embs = self.encoder(batch)
        # ② 어디의 Gaussian을 어떻게 바꿀지: UV CNN -> raw geometry/appearance map.
        # ③ posed 표면과 합쳐 Gaussian을 배치하는 fixed decode는 08.forward_frame_batch -> 05에 있다.
        return self.decoder(uv_input, embs)


def parameter_batch(scene, ids, device):
    index = torch.as_tensor(ids, dtype=torch.long)
    return {k: scene["parameters"][k][index].to(device) for k in
            ("expr", "rotation", "translation", "neck_pose", "jaw_pose", "eyes_pose")}


def load_config(path=None):
    return json.loads((path if path else L03.ROOT / "avatar_config.json").read_text())


def fetch_reference(root):
    """검증용 고정 revision의 공개 source만 cache에 저장한다. Weight는 가져오지 않는다."""
    import urllib.request
    for name, expected in UPSTREAM_SHA256.items():
        path = root / name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            continue
        url = f"https://raw.githubusercontent.com/kaist-ami/ELITE/{UPSTREAM_REVISION}/{name}"
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"원본 source hash 불일치: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    print("원본 source cache:", root)


def reference_model(config, root):
    """원본 파일의 AST를 독립 namespace로 읽어 별도의 구현으로 비교한다.

    전체 model import에 따른 renderer 의존성은 여기서 필요하지 않다.
    Source hash를 확인하고 선택한 정의의 본문을 수정하지 않은 채 실행한다.
    """
    for name, expected in UPSTREAM_SHA256.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"{path}가 없다. 먼저 --fetch-reference를 실행하세요.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"원본 source가 변경됐다: {path}")
    common = {name: globals()[name] for name in
              ("np", "torch", "th", "nn", "F", "thf", "silu", "copy", "inspect", "_pair",
               "remove_weight_norm", "WeightNorm", "Any", "Dict", "List", "Optional", "Tuple", "Type", "Union")}

    def load(path, names, extra=None):
        tree = ast.parse((root / path).read_text())
        chosen = []
        for node in tree.body:
            key = node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else None
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                key = node.targets[0].id
            if key in names:
                chosen.append(node)
        if len(chosen) != len(names):
            raise ValueError(f"필요한 원본 정의가 없다: {path}")
        scope = dict(common, __name__="elite_reference_" + Path(path).stem)
        scope.update(extra or {})
        exec(compile(ast.Module(body=chosen, type_ignores=[]), str(root / path), "exec"), scope)
        return scope

    layers = load("src/nn/layers.py", REFERENCE_NODES["src/nn/layers.py"])
    ref_la = SimpleNamespace(**{k: layers[k] for k in ("LinearWN", "Conv2dWNUB", "ConvTranspose2dWNUB", "glorot")})
    song = load("src/nn/unet_gs.py", REFERENCE_NODES["src/nn/unet_gs.py"])
    heads = load("src/nn/unet.py", ["UNetWBConcat"], {"la": ref_la})
    from torchvision.transforms import transforms
    models = load("src/models/mesh_unet.py", ["ExprEncoder", "MeshUNetPrimDecoder", "MeshUNetPriorModel2DGS"], {
        "la": ref_la, "make_linear": layers["make_linear"], "SongUNet": song["SongUNet"],
        "UNetWBConcat": heads["UNetWBConcat"], "rotvec_to_unitquat": rotvec_to_unitquat,
        "transforms": transforms,
    })
    return models["MeshUNetPriorModel2DGS"](official_config(config), device="cpu", is_train=True)


def check_parity(config, reference, size, device):
    """원본과 같은 초기값으로 eval/train forward와 모든 파라미터의 gradient를 비교한다."""
    cfg = copy.deepcopy(config)
    cfg["uv_size"] = size
    torch.manual_seed(cfg["seed"])
    model = Mesh2Gaussian(cfg).to(device)
    torch.manual_seed(cfg["seed"])
    original = reference_model(cfg, reference).to(device)
    ours, theirs = model.state_dict(), original.state_dict()
    if list(ours) != list(theirs):
        raise AssertionError("공식 network와 state_dict의 key/order가 다르다")
    for name in ours:
        torch.testing.assert_close(ours[name], theirs[name], rtol=0, atol=0, msg=f"initial state: {name}")
    generator = torch.Generator().manual_seed(cfg["seed"] + 1)
    uv = torch.randn(1, 6, size, size, generator=generator).to(device) * .2
    parameters = {name: torch.randn(1, width, generator=generator).to(device)*.1 for name, width in
                  (("expr", 100), ("rotation", 3), ("translation", 3), ("neck_pose", 3), ("jaw_pose", 3), ("eyes_pose", 6))}
    targets = [torch.randn(1, width, size, size, generator=generator).to(device) for width in (13, 3)]
    report = {"revision": UPSTREAM_REVISION, "device": str(device),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "fixture_uv_size": size,
              "state_tensors": len(ours), "initial_state": "exact", "modes": {}}
    for mode in ("eval", "train"):
        model.train(mode == "train")
        original.train(mode == "train")
        records = []
        for candidate, is_original in ((model, False), (original, True)):
            candidate.zero_grad(set_to_none=True)
            x = uv.detach().clone().requires_grad_(True)
            p = {name: value.detach().clone().requires_grad_(True) for name, value in parameters.items()}
            torch.manual_seed(cfg["seed"] + 2)  # train 모드의 dropout mask도 일치시킨다.
            if is_original:
                batch = {"flame_expr": p["expr"], "flame_rot": p["rotation"], "flame_trans": p["translation"],
                         "flame_neck": p["neck_pose"], "flame_jaw": p["jaw_pose"], "flame_eyes": p["eyes_pose"]}
                features = candidate.decoder.unet(x, candidate.encoder(batch))
                outputs = (candidate.decoder.geo_enhancenet(features), candidate.decoder.app_enhancenet(features))
            else:
                outputs = candidate(x, p)
            loss = sum((output * target).mean() for output, target in zip(outputs, targets))
            loss.backward()
            records.append((tuple(output.detach() for output in outputs),
                            {name: None if value.grad is None else value.grad.detach().clone()
                             for name, value in candidate.named_parameters()},
                            {"uv_input": x.grad.detach().clone(), **{name: value.grad.detach().clone() for name, value in p.items()}}))
        left, right = records
        maximum = 0.
        compared = 0
        for a, b in zip(left[0], right[0]):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
            maximum = max(maximum, float((a-b).abs().max()))
        for name, gradient in left[1].items():
            expected = right[1][name]
            if gradient is None or expected is None:
                if gradient is not None or expected is not None:
                    raise AssertionError(f"gradient presence: {name}")
                continue
            if not torch.isfinite(gradient).all() or not torch.isfinite(expected).all():
                raise AssertionError(f"non-finite gradient: {name}")
            torch.testing.assert_close(gradient, expected, rtol=1e-5, atol=1e-6, msg=f"gradient: {name}")
            compared += 1
        for name, gradient in left[2].items():
            torch.testing.assert_close(gradient, right[2][name], rtol=1e-5, atol=1e-6, msg=f"input gradient: {name}")
        report["modes"][mode] = {"max_output_difference": maximum, "compared_parameter_gradients": compared}
        print("공식 network parity:", mode, report["modes"][mode], flush=True)
        del records, left, right
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    L03.add_arguments(parser)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--fetch-reference", action="store_true", help="고정 revision의 공개 source를 cache에 준비")
    parser.add_argument("--check-parity", action="store_true", help="원본과 초기 state/forward/backward 비교")
    parser.add_argument("--check-training-step", action="store_true", help="실제 RGB/전체 loss/backward/optimizer 한 step 확인")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--parity-size", type=int, default=64, help="비교용 UV size. 작은 fixture에서도 layer 채널 수와 6단 구조는 유지")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.check_parity and args.check_training_step:
        parser.error("parity와 실제 training step은 각각 실행하세요.")
    if args.fetch_reference:
        fetch_reference(args.reference)
        if not (args.check_parity or args.check_training_step):
            return
    if args.check_parity:
        report = check_parity(config, args.reference, args.parity_size, torch.device(args.device))
        L03.save_json(L03.output_dir(6, args.sequence, args.run)/"network_parity.json", report)
        return
    scene = L03.read_scene(args.sequence, args.run)
    uv = L03.lesson(5).read_uv(scene)
    if config["uv_size"] != uv["uv_size"]:
        raise ValueError("config와 04의 UV resolution을 맞추세요.")
    if args.check_training_step:
        check_training_step(config, scene, uv, torch.device(args.device))
        return
    torch.manual_seed(config["seed"])
    model = Mesh2Gaussian(config).to(args.device).eval()
    # model.eval(): dropout OFF. torch.no_grad()와 역할이 다르며 둘 다 optimizer 업데이트는 하지 않는다.
    shape_log, feature_log, handles = [], {}, []
    size = config["uv_size"]
    middle = f"decoder.unet.dec.{size//32}x{size//32}_in0"
    def record(name):
        def hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                shape_log.append({"layer": name, "shape": list(output.shape)})
                if name == middle:
                    feature_log[name] = output[0, :4].detach().cpu()
        return hook
    for name, module in model.named_modules():
        if isinstance(module, (UNetBlock, Conv2dUB, ConvTranspose2dUB, ExprEncoder)):
            handles.append(module.register_forward_hook(record(name)))
    with torch.no_grad():
        # RGB 채널 먼저, XYZ 채널 뒤. 이 6채널 입력은 05에서 수동으로 만든 13+3채널 map과 다르다.
        input_uv = torch.cat((uv["texture"]*2-1, uv["normalized_xyz"]), 0)[None].to(args.device)
        frame_id = scene["split"]["train"][0]
        geo, app = model(input_uv, parameter_batch(scene, [frame_id], args.device))
        # Random network의 raw 출력이다. 05의 수동 Gaussian 초기값을 network target으로 학습하지 않는다.
    for handle in handles:
        handle.remove()
    out = L03.output_dir(6, args.sequence, args.run)
    L03.save_tensor(out/"initial_network_outputs.pt", {"geometry": geo.cpu(), "appearance": app.cpu(),
                                                     "bottleneck_features": feature_log})
    for feature in feature_log.values():
        # 각 채널을 따로 min/max normalize한다. 색은 XYZ/RGB나 attention 확률이 아니라 임의 feature 값이다.
        tiles = []
        for channel in feature:
            normalized = (channel-channel.min())/(channel.max()-channel.min()).clamp_min(1e-8)
            tiles.append(normalized[None].expand(3, -1, -1))
        image = F.interpolate(torch.cat(tiles, -1)[None], scale_factor=8, mode="nearest")[0]
        L03.save_image(out/"initial_bottleneck_features.png", image)
    L03.save_json(out/"network_shapes.json", {
        "revision": UPSTREAM_REVISION, "format": MODEL_FORMAT,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "geometry_output": list(geo.shape), "appearance_output": list(app.shape), "layers": shape_log,
        "scope": "official learned network; scratch training; fixed decoding/preprocessing/loss in separate lessons",
    })
    for row in shape_log:
        print(row["layer"], row["shape"])
    print("06 완료:", out, "| 원본 초기화의 network forward. 학습은 08에서 진행한다.")


def check_training_step(config, scene, uv, device):
    """08의 실제 network -> 05 decode -> 2DGS -> 07 loss -> optimizer를 한 번 확인한다.

    전체 학습을 진행하거나 checkpoint를 바꾸지 않는다. 초기 학습률은 08 warmup의 첫 값.
    전후에 같은 dropout seed를 사용해 optimizer로 인한 변화를 비교한다.
    """
    L05, L07, L08 = L03.lesson(5), L03.lesson(7), L03.lesson(8)
    if device.type != "cuda":
        raise ValueError("2DGS renderer의 실제 training step 검증은 --device cuda:0으로 실행하세요.")
    L08.validate_config(config, uv)
    L08.seed_all(config["seed"])
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model = Mesh2Gaussian(config).to(device).train()
    surface = L05.SurfaceMap(scene, uv, device)
    dataset = L07.VideoFrames(scene, config["image_max_side"])
    objective = L07.AvatarLoss(config, uv, device)
    ids = [scene["split"]["train"][0]]
    target = dataset.batch(ids, device)
    background = torch.ones(3, device=device)
    expected = L07.composite(target["rgb"], target["alpha"], background)[0]
    training = config["training"]
    rate = training["lr"] / max(1, training["warmup_steps"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=training["weight_decay"])
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    torch.manual_seed(config["seed"] + 17)
    gaussians = L08.forward_frame_batch(model, surface, scene, ids, device)
    for key in ("geometry_map", "appearance_map"):
        gaussians[key].retain_grad()
    rendered = L08.render_batch(gaussians, dataset, device, background)
    total, raw, weighted = objective(rendered, target, gaussians, background, step=0)
    if not torch.isfinite(total):
        raise FloatingPointError("초기 실제 training loss가 유한하지 않다.")
    initial_image = rendered[0]["rgb"].detach().clone()
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    map_gradients = {}
    for key in ("geometry_map", "appearance_map"):
        gradient = gaussians[key].grad
        if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
            raise AssertionError(f"실제 renderer에서 {key}까지 gradient 연결 실패")
        map_gradients[key] = gradient.abs().mean((0, 2, 3)).detach().cpu().tolist()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"], error_if_nonfinite=True)
    modules = L08.gradient_report(model)
    if not all(np.isfinite(value) and value > 0 for value in modules.values()):
        raise AssertionError(f"모든 학습 모듈로 유한한 nonzero gradient가 필요하다: {modules}")
    optimizer.step()
    changed = sum(not torch.equal(value.detach(), before[name]) for name, value in model.named_parameters())
    if changed == 0:
        raise AssertionError("Optimizer가 network 파라미터를 바꾸지 못했다.")
    torch.manual_seed(config["seed"] + 17)
    with torch.no_grad():
        updated = L08.forward_frame_batch(model, surface, scene, ids, device)
        result = L08.render_batch(updated, dataset, device, background)
        after_loss, _, _ = objective(result, target, updated, background, step=0)
    report = {
        "revision": UPSTREAM_REVISION, "format": MODEL_FORMAT,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu": torch.cuda.get_device_name(device),
        "frame_ids": ids, "uv_size": config["uv_size"], "image_size": [dataset.h, dataset.w],
        "loss_before": float(total.detach()), "loss_after": float(after_loss),
        "raw_loss": {key: float(value.detach()) for key, value in raw.items()},
        "weighted_loss": {key: float(value.detach()) for key, value in weighted.items()},
        "gradient_norm_before_clip": float(norm), "gradient_by_module": modules,
        "map_gradient_per_channel": map_gradients, "updated_parameter_tensors": changed,
        "visible_gaussians": int((rendered[0]["radii"] > 0).sum()),
        "gpu_peak_gb": torch.cuda.max_memory_allocated(device)/1024**3,
        "scope": "one real training step; all configured losses at step 0; no convergence claim",
    }
    out = L03.output_dir(6, scene["sequence"], scene["run"])
    L03.save_json(out/"training_step_check.json", report)
    L03.save_image(out/"training_step_before.png", initial_image)
    L03.save_image(out/"training_step_after.png", result[0]["rgb"])
    L03.save_image(out/"training_step_target.png", expected)
    print("실제 training step 확인:", report, flush=True)


if __name__ == "__main__":
    main()
