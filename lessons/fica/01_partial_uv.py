"""01 — 한 장의 RGB에서 UV / canonical XYZ / normal을 예측하고 partial UV를 만든다.

세 task는 독립적인 Sapiens-1B encoder를 fine-tune한다(FiCA C.1).
UV/XYZ: pre-trained encoder + depth-style decoder. Normal: official normal checkpoint.
큰 OpenMMLab framework 대신 같은 state_dict 이름/연산의 PyTorch port를 사용한다.
Encoder/head를 더미 CNN으로 대체하지 않는다. 모델 크기는 smoke에서도 실제 1B다.

CUDA_VISIBLE_DEVICES=0 python 01_partial_uv.py --stage train --task uv
CUDA_VISIBLE_DEVICES=0 python 01_partial_uv.py --stage train --task xyz
CUDA_VISIBLE_DEVICES=0 python 01_partial_uv.py --stage train --task normal
CUDA_VISIBLE_DEVICES=0 python 01_partial_uv.py --stage predict
CUDA_VISIBLE_DEVICES=0 python 01_partial_uv.py --stage finetune-uv --steps 3000

기본 train run=crop, finetune-uv/predict run=uv_repair. Validation은 0/100/200.jpg.
학습 화면은 tqdm bar만 갱신한다.
기존 checkpoint 평가: --stage validate --task uv --checkpoint checkpoints/01/uv.pt
"""

import importlib.util
from pathlib import Path
import math
import json

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as recompute
from tqdm.auto import tqdm

spec = importlib.util.spec_from_file_location("fica_00", Path(__file__).with_name("00_prepare_nersemble.py"))
L00 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L00)


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        batch, tokens, width = x.shape
        q, k, v = self.qkv(x).reshape(batch, tokens, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(batch, tokens, width))


class FeedForward(nn.Module):
    def __init__(self, width):
        super().__init__()
        # OpenMMLab FFN의 checkpoint key: ffn.layers.0.0 / ffn.layers.1.
        self.layers = nn.Sequential(nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Dropout(0)),
                                    nn.Linear(width * 4, width), nn.Dropout(0))

    def forward(self, x):
        return self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(width, eps=1e-6)
        self.attn = Attention(width, heads)
        self.ln2 = nn.LayerNorm(width, eps=1e-6)
        self.ffn = FeedForward(width)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class SapiensEncoder(nn.Module):
    """Sapiens-1B: patch16, dim1536, 40 blocks, 24 heads, LN eps=1e-6.

    Position embedding 크기는 checkpoint에서 읽고 입력 해상도로 bicubic 보간한다.
    모든 weight를 strict=True로 읽는다. cls-token이 있는 pretrain도 지원한다.
    """
    def __init__(self, width=1536, depth=40, heads=24, pos_tokens=4096, cls_token=False, padding=0):
        super().__init__()
        self.width, self.padding = width, padding
        self.patch_embed = nn.Module()
        self.patch_embed.projection = nn.Conv2d(3, width, 16, stride=16, padding=padding)
        self.pos_embed = nn.Parameter(torch.zeros(1, pos_tokens, width))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width)) if cls_token else None
        self.layers = nn.ModuleList([EncoderBlock(width, heads) for _ in range(depth)])
        self.ln1 = nn.LayerNorm(width, eps=1e-6)
        self.checkpoint_blocks = True

    def forward(self, rgb):
        mean = rgb.new_tensor([.485, .456, .406])[None, :, None, None]
        std = rgb.new_tensor([.229, .224, .225])[None, :, None, None]
        x = self.patch_embed.projection((rgb - mean) / std)
        batch, _, height, width = x.shape
        pos = self.pos_embed[:, 1:] if self.cls_token is not None else self.pos_embed
        count = pos.shape[1]
        side = int(math.sqrt(count))
        ph, pw = (side, side) if side * side == count else (64, 48)
        if ph * pw != count:
            raise ValueError(f"알 수 없는 Sapiens positional grid {count}")
        pos = pos.reshape(1, ph, pw, self.width).permute(0, 3, 1, 2)
        pos = F.interpolate(pos.float(), size=(height, width), mode="bicubic", align_corners=False).to(x.dtype)
        x = x.flatten(2).transpose(1, 2) + pos.flatten(2).transpose(1, 2)
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(batch, -1, -1) + self.pos_embed[:, :1], x), 1)
        for block in self.layers:
            x = recompute(block, x, use_reentrant=False) if self.training and self.checkpoint_blocks else block(x)
        x = self.ln1(x)
        if self.cls_token is not None:
            x = x[:, 1:]
        return x.transpose(1, 2).reshape(batch, self.width, height, width)


def official_state(key):
    # Pickle은 여기서 명시한 공식 revision의 초기화 checkpoint에만 사용한다.
    path = L00.asset_path(key)
    source = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    return source.get("state_dict", source)


def encoder_from_state(state, device, padding=0, remove_cls=False):
    values = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}
    if not values:
        values = dict(state)  # 공식 pretrain은 backbone prefix 없이 배포된다.
    if remove_cls and "cls_token" in values:
        values.pop("cls_token")
        values["pos_embed"] = values["pos_embed"][:, 1:]
    # meta 초기화는 1B random weight를 먼저 만들었다가 덮는 메모리 낭비를 피한다.
    with torch.device("meta"):
        encoder = SapiensEncoder(pos_tokens=values["pos_embed"].shape[1], cls_token="cls_token" in values, padding=padding)
    encoder.load_state_dict(values, strict=True, assign=True)
    return encoder.to(device)


def pretrained_encoder(device):
    encoder = encoder_from_state(official_state("sapiens_encoder"), device)
    encoder.eval().requires_grad_(False)
    return encoder


class DenseHead(nn.Module):
    """공식 depth: 4x384 deconv + 2x384 conv, normal: 3x768 + 3x768.

    InstanceNorm(affine=False), SiLU, bias 없는 ConvTranspose2d를 그대로 따른다.
    UV/XYZ output activation은 적용하지 않는다. UV의 범위 밖 예측은 splat 때 제외한다.
    """
    def __init__(self, output_channels, normal=False):
        super().__init__()
        width, stages, convs = (768, 3, 3) if normal else (384, 4, 2)
        layers, incoming = [], 1536
        for _ in range(stages):
            layers.extend([nn.ConvTranspose2d(incoming, width, 4, 2, 1, bias=False), nn.InstanceNorm2d(width), nn.SiLU(inplace=True)])
            incoming = width
        self.deconv_layers = nn.Sequential(*layers)
        layers = []
        for _ in range(convs):
            layers.extend([nn.Conv2d(width, width, 1), nn.InstanceNorm2d(width), nn.SiLU(inplace=True)])
        self.conv_layers = nn.Sequential(*layers)
        self.conv_seg = nn.Conv2d(width, output_channels, 1)

    def forward_features(self, features):
        return self.conv_layers(self.deconv_layers(features))

    def forward(self, features):
        return self.conv_seg(self.forward_features(features))


class DensePredictor(nn.Module):
    def __init__(self, task, device, input_size=(1024,768)):
        super().__init__()
        if task not in ("uv", "xyz", "normal"):
            raise ValueError(task)
        self.task = task
        self.input_size = tuple(input_size)
        if task == "normal":
            state = official_state("sapiens_normal")
            self.backbone = encoder_from_state(state, device, padding=2)
            self.decode_head = DenseHead(3, normal=True).to(device)
            self.decode_head.load_state_dict({k[len("decode_head."):]: v for k, v in state.items() if k.startswith("decode_head.")}, strict=True)
        else:
            self.backbone = encoder_from_state(official_state("sapiens_encoder"), device, remove_cls=True)
            self.decode_head = DenseHead(2 if task == "uv" else 3).to(device)

    def forward(self, rgb):
        shape = rgb.shape[-2:]
        resized = F.interpolate(rgb, size=self.input_size, mode="bilinear", align_corners=False)
        result = self.decode_head(self.backbone(resized))
        result = F.interpolate(result, size=shape, mode="bilinear", align_corners=False)
        return F.normalize(result.float(), dim=1, eps=1e-6) if self.task == "normal" else result.float()


REPAIR_VARIANT = "uv_validity_fp32_v1"


def validity_head():
    """공유 UV feature -> foreground logit. FiCA 공개 구조가 아닌 우리의 추가 분기.

    UV 좌표가 맞는지의 확률이 아니라, 이 픽셀이 UV 학습 대상 표면인지 예측한다.
    처음에는 보수적으로 미관측으로 두고 BCE/Dice로 학습한다.
    """
    head = nn.Sequential(nn.Conv2d(384, 64, 3, padding=1), nn.SiLU(), nn.Conv2d(64, 1, 1))
    nn.init.zeros_(head[-1].weight)
    nn.init.constant_(head[-1].bias, -2.)
    return head


class UVValidityPredictor(DensePredictor):
    """기존 2ch UV 모델을 그대로 가져오고 validity 분기만 새로 초기화한다.

    Backbone은 기존 AMP 정책을 따르고 decoder 두 분기와 좌표 보간은 FP32다.
    forward는 기존과 같은 UV tensor를 반환한다. 추가 학습은 predict()로 logit도 받는다.
    """
    def __init__(self, base):
        nn.Module.__init__(self)
        if base.task != "uv" or isinstance(base, UVValidityPredictor):
            raise ValueError("기존 UV 모델로 초기화하세요. Repair checkpoint는 --resume을 사용하세요.")
        self.task, self.input_size = "uv", base.input_size
        self.backbone, self.decode_head = base.backbone, base.decode_head
        self.validity_head = validity_head().to(next(base.parameters()).device)

    def predict(self, rgb):
        shape = rgb.shape[-2:]
        resized = F.interpolate(rgb, size=self.input_size, mode="bilinear", align_corners=False)
        features = self.backbone(resized)
        with torch.autocast(rgb.device.type, enabled=False):
            hidden = self.decode_head.forward_features(features.float())
            uv = self.decode_head.conv_seg(hidden)
            logit = self.validity_head(hidden)
            uv = F.interpolate(uv, size=shape, mode="bilinear", align_corners=False)
            logit = F.interpolate(logit, size=shape, mode="bilinear", align_corners=False)
        return uv, logit

    def forward(self, rgb):
        return self.predict(rgb)[0]


def prediction_loss(task, predicted, sample, mean, std):
    labels = sample["labels"]
    target = labels[task][None]
    mask = labels["valid"].float()[None]
    if task == "normal":
        error = 1 - (F.normalize(predicted, dim=1) * target).sum(1, keepdim=True)
    elif task == "xyz":
        # 학습 안정화를 위한 좌표 단위 정규화. 저장되는 prediction은 다시 world unit으로 복원.
        error = (predicted - (target - mean) / std).abs().mean(1, keepdim=True)
    else:
        error = (predicted - target).abs().mean(1, keepdim=True)
    if not mask.any():
        raise ValueError("label foreground가 없다. 00 mesh/camera 정합을 확인하세요.")
    return (error * mask).sum() / mask.sum()


INPUT_TRANSFORM = "aspect_crop_v1"


def prepare_input(sample, input_size, augment=False):
    """RGB와 image-space labels를 함께 자른다. UV/XYZ 값 자체는 바뀌지 않는다.

    예: 512x351 -> 중앙 468x351 -> Sapiens 1024x768. 두 단계의 H:W가
    모두 4:3이므로 얼굴 비율이 유지된다. 학습만 crop 위치/크기를 무작위로 고른다.
    Labels는 보간하지 않는다. DensePredictor가 출력을 crop 해상도로 되돌려 비교한다.
    """
    height, width = sample["rgb"].shape[-2:]
    divisor = math.gcd(*input_size)
    ah, aw = input_size[0] // divisor, input_size[1] // divisor
    unit = min(height // ah, width // aw)
    if unit < 1:
        raise ValueError("입력이 Sapiens crop 비율보다 작다.")
    if augment:
        unit = max(1, round(unit * float(torch.empty(()).uniform_(.8, 1.))))
    h, w = ah * unit, aw * unit
    top = int(torch.randint(height-h+1, ())) if augment else (height-h)//2
    left = int(torch.randint(width-w+1, ())) if augment else (width-w)//2
    def crop(value):
        return value[..., top:top+h, left:left+w]
    result = dict(sample, rgb=crop(sample["rgb"]), alpha=crop(sample["alpha"]),
                  labels={key: crop(value) for key, value in sample["labels"].items()},
                  crop={"top": top, "left": left, "height": h, "width": w,
                        "source_hw": [height, width]})
    # Camera/keypoints도 crop 좌표계로 옮긴다. Partial UV는 원본 화면 기준이므로 제거한다.
    result.pop("partial_gt", None)
    if "K" in sample:
        result["K"] = sample["K"].clone()
        result["K"][0, 2] -= left
        result["K"][1, 2] -= top
    if "keypoints" in sample:
        points = sample["keypoints"] - sample["keypoints"].new_tensor([left, top])
        result["keypoints"] = points
        result["keypoint_valid"] = sample["keypoint_valid"] & (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
    if augment:
        rgb = result["rgb"]
        mean = rgb.mean((-2, -1), keepdim=True)
        contrast = float(torch.empty(()).uniform_(.9, 1.1))
        brightness = float(torch.empty(()).uniform_(-.05, .05))
        result["rgb"] = ((rgb-mean)*contrast+mean+brightness).clamp(0, 1)
    return result


def run_paths(config, task, run_name="crop"):
    if not run_name or Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("--run-name에는 폴더 이름 하나만 지정하세요.")
    root = L00.work_root(config)
    return root / "outputs/01" / run_name / task, root / "checkpoints/01" / run_name / f"{task}.pt"


def load_predictor(path, task, config, data, device):
    """기존 모델의 weight만 읽는다. Optimizer/step을 이어받는 --resume과 구별한다.

    mmap으로 수 GB Adam state를 불필요하게 메모리에 올리지 않는다. Meta 생성은
    DensePredictor와 같은 클래스/모든 state key를 사용하며 strict load를 유지한다.
    """
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if state["format"] != "fica_checkpoint_v1" or state["extra"].get("task") != task:
        raise ValueError("checkpoint format/task 불일치")
    if state["data_sha256"] != L00.sha256(data.path / "manifest.json"):
        raise ValueError("checkpoint dataset 불일치")
    values = state["model"]
    with torch.device("meta"):
        repaired = any(key.startswith("validity_head.") for key in values)
        if repaired and (task != "uv" or state["extra"].get("variant") != REPAIR_VARIANT):
            raise ValueError("지원하지 않는 UV validity checkpoint")
        kind = UVValidityPredictor if repaired else DensePredictor
        model = kind.__new__(kind)
        nn.Module.__init__(model)
        model.task, model.input_size = task, tuple(config["sapiens"]["input_size"])
        model.backbone = SapiensEncoder(pos_tokens=values["backbone.pos_embed"].shape[1],
                                        cls_token="backbone.cls_token" in values,
                                        padding=2 if task == "normal" else 0)
        model.decode_head = DenseHead(2 if task == "uv" else 3, normal=task == "normal")
        if repaired:
            model.validity_head = validity_head()
    model.load_state_dict(values, strict=True, assign=True)
    metadata = {"checkpoint": str(Path(path).resolve()), "source_step": state["step"],
                "source_code_hashes": state["code_hashes"], "source_config": state["config"],
                "source_input_transform": state["extra"].get("input_transform", "legacy_resize"),
                "variant": state["extra"].get("variant", "legacy"), "extra": state["extra"]}
    return model.to(device), metadata


def validation_rows(data, limit):
    rows = data.samples("validation")
    # 항상 같은 순서/같은 frame. Train sample로 대신 평가하지 않는다.
    return rows[:limit] if limit else rows


def heatmap(error, valid, maximum):
    ratio = torch.nan_to_num(error / maximum, nan=1., posinf=1., neginf=0.).clamp(0, 1)
    return torch.cat((ratio, 1-(2*ratio-1).abs(), 1-ratio), 0) * valid


def save_validation_panel(path, model, sample, predicted, mean, std, uv_size, step, row, metrics):
    """매 step 같은 척도를 사용한다. 표시 패널도 letterbox로 종횡비를 보존한다."""
    valid = sample["labels"]["valid"]
    target = sample["labels"][model.task]
    rgb = F.interpolate(sample["rgb"][None], size=model.input_size, mode="bilinear", align_corners=False)[0]
    images, titles = [rgb], [f"Actual input {model.input_size[0]}x{model.input_size[1]}"]
    if model.task == "uv":
        def uv_color(value):
            color = torch.cat((value.clamp(0, 1), torch.zeros_like(value[:1])), 0)
            outside = (~torch.isfinite(value).all(0, keepdim=True)) | (value < 0).any(0, keepdim=True) | (value > 1).any(0, keepdim=True)
            return torch.where(outside.expand_as(color), color.new_tensor([1, 0, 1])[:, None, None], color) * valid
        error = (predicted-target).square().sum(0, keepdim=True).sqrt()*uv_size
        images += [uv_color(target), uv_color(predicted), heatmap(error, valid, 32)]
        titles += ["GT UV: R=u, G=v", "Pred UV (magenta=out of range)", "UV error: blue=0, red>=32 texels"]
        # UV 대응만의 진단이다. XYZ/normal은 미입력이며 완성된 10ch 예측이 아니다.
        zeros = torch.zeros_like(sample["rgb"])
        for name, uv in (("GT", target), ("Pred", predicted)):
            partial, _ = L00.splat_partial_uv(sample["rgb"], uv, zeros, zeros, valid, uv_size)
            images.append(partial[:3])
            titles.append(f"{name} UV RGB splat (valid mask)")
    elif model.task == "xyz":
        world = predicted*std[0]+mean[0]
        color = lambda value: (((value-mean[0])/std[0])/6+.5).clamp(0, 1)*valid
        error = (world-target).square().sum(0, keepdim=True).sqrt()
        scale = float(std.mean())*.25
        images += [color(target), color(world), heatmap(error, valid, scale)]
        titles += ["GT XYZ (mean +/- 3 std)", "Pred XYZ (same color scale)", f"XYZ error: blue=0, red>={scale:.4g}"]
    else:
        error = torch.rad2deg(torch.acos((predicted*target).sum(0, keepdim=True).clamp(-1, 1)))
        images += [(target*.5+.5)*valid, (predicted*.5+.5)*valid, heatmap(error, valid, 45)]
        titles += ["GT camera-space normal", "Pred camera-space normal", "Angle error: blue=0, red>=45 deg"]
    width, height, header, columns = 300, 400, 62, 3
    panel_rows = math.ceil(len(images)/columns)
    canvas = Image.new("RGB", (columns*width, header+panel_rows*(height+28)), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), f"{model.task} | optimizer step {step} | held-out {Path(row['file']).stem}", fill="white")
    draw.text((10, 28), " | ".join(f"{key}={value:.5g}" for key, value in metrics.items()), fill="white")
    draw.text((10, 45), "Fixed center crop; same frame and error scale throughout training", fill="white")
    for index, (value, title) in enumerate(zip(images, titles)):
        array = (torch.nan_to_num(value.detach().float()).clamp(0, 1).cpu().permute(1, 2, 0).numpy()*255).round().astype(np.uint8)
        tile = Image.fromarray(array)
        scale = min(width/tile.width, height/tile.height)
        tile = tile.resize((round(tile.width*scale), round(tile.height*scale)), Image.Resampling.BILINEAR)
        x, y = (index % columns)*width, header+(index//columns)*(height+28)
        draw.text((x+5, y+6), title, fill="white")
        canvas.paste(tile, (x+(width-tile.width)//2, y+28+(height-tile.height)//2))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=95)


@torch.no_grad()
def validate(model, data, device, out, step, mean, std, rows):
    """Held-out pixels의 loss/물리 오차를 집계하고 첫 고정 frame만 JPEG로 저장한다."""
    if not rows:
        raise ValueError("validation split이 없다. 00의 split을 먼저 준비하세요.")
    was_training = model.training
    model.eval()
    totals, count, preview = {}, 0, None
    try:
        for row in rows:
            sample = prepare_input(data.sample(row, device), model.input_size)
            valid = sample["labels"]["valid"]
            pixels = int(valid.sum())
            if not pixels:
                raise ValueError(f"validation foreground가 없다: {row['file']}")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted = model(sample["rgb"][None])
            if not torch.isfinite(predicted).all():
                raise FloatingPointError("validation prediction에 NaN/Inf가 있다.")
            loss = prediction_loss(model.task, predicted, sample, mean, std)
            predicted = predicted[0]
            target = sample["labels"][model.task]
            values = {"loss": float(loss)}
            if model.task == "uv":
                distance = (predicted-target).square().sum(0, keepdim=True).sqrt()*data.manifest["config"]["uv_size"]
                in_range = ((predicted >= 0) & (predicted <= 1)).all(0, keepdim=True)
                values.update(uv_error_texels=float(distance[valid].mean()), uv_in_range=float(in_range[valid].float().mean()))
            elif model.task == "xyz":
                distance = (predicted*std[0]+mean[0]-target).square().sum(0, keepdim=True).sqrt()
                values["xyz_error_world"] = float(distance[valid].mean())
            else:
                angle = torch.rad2deg(torch.acos((predicted*target).sum(0, keepdim=True).clamp(-1, 1)))
                values["normal_error_degrees"] = float(angle[valid].mean())
            for key, value in values.items():
                totals[key] = totals.get(key, 0.)+value*pixels
            count += pixels
            if preview is None:
                preview = (sample, predicted, row, values)
        metrics = {key: value/count for key, value in totals.items()}
        directory = out / "validation"
        save_validation_panel(directory/f"{step}.jpg", model, preview[0], preview[1], mean, std,
                              data.manifest["config"]["uv_size"], step, preview[2], preview[3])
        L00.append_log(directory/"metrics.jsonl", {"step": step, **metrics, "pixels": count,
                       "frames": [row["file"] for row in rows], "preview_frame": preview[2]["file"],
                       "input_transform": INPUT_TRANSFORM, "crop": preview[0]["crop"],
                       "scope": data.manifest["scope"]})
        return metrics
    finally:
        model.train(was_training)


@torch.no_grad()
def export_predictions(model, data, device, mean, std, run_name, step):
    """원본 화면 크기를 유지한다. Crop 밖 UV=-1은 splat에서 제외된다."""
    model.eval()
    for row in tqdm(data.manifest["samples"], desc=f"01 {model.task} predictions", dynamic_ncols=True, mininterval=1):
        original = data.sample(row, device)
        sample = prepare_input(original, model.input_size)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted = model(sample["rgb"][None])
        if model.task == "xyz":
            predicted = predicted*std+mean
        crop = sample["crop"]
        full = predicted.new_full((predicted.shape[1], *original["rgb"].shape[-2:]), -1 if model.task == "uv" else 0)
        y, x, h, w = (crop[key] for key in ("top", "left", "height", "width"))
        full[:, y:y+h, x:x+w] = predicted[0]
        path = data.path / "predictions" / model.task / f"{Path(row['file']).stem}.pt"
        L00.save_tensor(path, full.cpu())
        L00.save_json(path.with_suffix(".json"), {"run": run_name, "step": step, "crop": crop, "input_transform": INPUT_TRANSFORM})


def train_task(config, data, task, device, steps, save=True, resume=False,
               val_every=100, val_frames=8, run_name="crop", init_from=None):
    out, checkpoint_path = run_paths(config, task, run_name)
    if val_every < 1 or val_frames < 0:
        raise ValueError("val_every >= 1, val_frames >= 0 이어야 한다.")
    rows = validation_rows(data, val_frames)
    if not rows:
        raise ValueError("validation split이 없다. 학습 전에 00의 split을 준비하세요.")
    if resume and init_from:
        raise ValueError("--resume과 --init-from은 함께 사용할 수 없다.")
    if save and not resume and checkpoint_path.exists() and config.get("profile") != "smoke":
        raise FileExistsError(f"기존 checkpoint 보존: {checkpoint_path}. 같은 실행은 --resume, 새 실행은 다른 --run-name을 사용하세요.")
    source = None
    if init_from or resume:
        model, source = load_predictor(init_from or checkpoint_path, task, config, data, device)
        if isinstance(model, UVValidityPredictor):
            raise ValueError("validity 모델의 학습은 --stage finetune-uv를 사용하세요.")
    else:
        model = DensePredictor(task, device, config["sapiens"]["input_size"])
    model.train()
    optimizer = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": config["sapiens"]["encoder_lr"]},
                                  {"params": model.decode_head.parameters(), "lr": config["sapiens"]["head_lr"]}], weight_decay=.01)
    start = L00.training_start(checkpoint_path, model, optimizer, config, data, True) if resume else 0
    if steps <= start:
        raise ValueError(f"--steps는 최종 step이다. 현재 step {start}보다 크게 지정하세요.")
    mean = torch.tensor(data.manifest["xyz_mean"], device=device)[None, :, None, None]
    std = torch.tensor(data.manifest["xyz_std"], device=device)[None, :, None, None]
    L00.save_json(out/"run.json", {"input_transform": INPUT_TRANSFORM, "run": run_name,
                  "val_every": val_every, "validation_frames": [row["file"] for row in rows],
                  "config": config, "code_hashes": L00.code_hashes(), "weights_initialization": source,
                  "start_step": start, "optimizer_reset": not resume})
    accumulation = config["training"]["grad_accumulation"]
    extra = {"task": task, "input_transform": INPUT_TRANSFORM, "run": run_name}
    with tqdm(total=steps, initial=start, desc=f"01 {task}", unit="step", dynamic_ncols=True, mininterval=1) as bar:
        recent_val = {}
        if start == 0 or start % val_every == 0:
            recent_val = validate(model, data, device, out, start, mean, std, rows)
            bar.set_postfix(val=f"{recent_val['loss']:.5f}", refresh=False)
        for step in range(start, steps):
            optimizer.zero_grad(set_to_none=True)
            mean_loss = 0.
            for _ in range(accumulation):
                sample = prepare_input(data.sample(data.training_row(), device), model.input_size,
                                       augment=config["sapiens"]["augmentation"])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    predicted = model(sample["rgb"][None])
                    loss = prediction_loss(task, predicted, sample, mean, std)
                (loss/accumulation).backward()
                mean_loss += float(loss.detach())/accumulation
            report = L00.gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["clip_grad"])
            optimizer.step()
            # 디버깅 수치는 파일에만 남긴다. 터미널에는 진행 bar 하나만 갱신한다.
            L00.append_log(out/"training.jsonl", {"step": step+1, "loss": mean_loss, "gradient": report, "scope": data.manifest["scope"]})
            if (step+1) % val_every == 0:
                recent_val = validate(model, data, device, out, step+1, mean, std, rows)
            bar.set_postfix(loss=f"{mean_loss:.5f}", val=f"{recent_val['loss']:.5f}" if recent_val else "pending", refresh=False)
            bar.update(1)
            if save and (step+1) % config["training"]["checkpoint_every"] == 0:
                L00.checkpoint(checkpoint_path, model, optimizer, config, data, step+1, extra)
    if save and steps % config["training"]["checkpoint_every"] != 0:
        L00.checkpoint(checkpoint_path, model, optimizer, config, data, steps, extra)
    export_predictions(model, data, device, mean, std, run_name, steps)
    del model, optimizer
    torch.cuda.empty_cache()




def resample_partial_uv(rgb, uv, xyz, normal, valid, size, context,
                        confidence=None, max_uv_edge=.03):
    """인접 image pixels를 UV 삼각형으로 rasterize -> image 좌표 -> grid_sample.

    Point splat과 달리 관측된 pixel cell의 내부를 채운다. 배경/가림/UV seam을
    가로질러 연결하지 않으며, GT mesh나 label을 입력으로 받지 않는다.
    confidence는 겹친 삼각형에서 사용할 foreground 확률이다(높은 쪽 우선).
    """
    import nvdiffrast.torch as dr
    if rgb.device.type != "cuda" or size < 1 or max_uv_edge <= 0:
        raise ValueError("CUDA tensor, size>=1, max_uv_edge>0이 필요하다.")
    height, width = rgb.shape[-2:]
    if min(height, width) < 2:
        raise ValueError("삼각형을 만들 수 있는 2x2 이상의 이미지가 필요하다.")
    expected = [(uv, 2), (xyz, 3), (normal, 3), (valid, 1)]
    if any(value.shape != (channels, height, width) for value, channels in expected):
        raise ValueError("RGB/UV/XYZ/normal/valid 해상도 불일치")
    features = torch.cat((rgb, normal, xyz), 0).float()
    point_valid = valid[0].bool() & torch.isfinite(features).all(0) & torch.isfinite(uv).all(0)
    point_valid &= (uv >= 0).all(0) & (uv <= 1).all(0)
    if confidence is not None:
        if confidence.shape != (1, height, width):
            raise ValueError("confidence shape 불일치")
        point_valid &= torch.isfinite(confidence[0])
    # grid_sample의 bilinear가 cell의 네 꼭짓점을 읽으므로 네 점 모두 유효해야 한다.
    cell = point_valid[:-1, :-1] & point_valid[:-1, 1:] & point_valid[1:, :-1] & point_valid[1:, 1:]
    a, b, c, d = uv[:, :-1, :-1], uv[:, :-1, 1:], uv[:, 1:, :-1], uv[:, 1:, 1:]
    cross = lambda x, y: x[0]*y[1]-x[1]*y[0]
    area0, area1 = cross(b-a, c-a), cross(d-b, c-b)
    edges = torch.stack([(b-a).norm(dim=0), (c-a).norm(dim=0), (d-b).norm(dim=0),
                         (d-c).norm(dim=0), (c-b).norm(dim=0)])
    continuous = torch.isfinite(edges).all(0) & (edges.amax(0) <= max_uv_edge)
    nonfolded = (area0*area1 > 0) & (area0.abs() > 1e-10) & (area1.abs() > 1e-10)
    candidate = cell & continuous & nonfolded
    # Chart별 반전은 허용하고, 주변 cell과 반대 방향인 국소 fold를 제외한다.
    signs = torch.where(candidate, area0.sign(), 0.)
    votes = F.avg_pool2d(signs[None, None], 3, stride=1, padding=1)[0, 0]
    keep = candidate & (area0*votes > 0)
    stats = {"source_pixels": int(point_valid.sum()), "source_cells": int(cell.sum()),
             "rejected_seam_cells": int((cell & ~continuous).sum()),
             "rejected_fold_cells": int((cell & continuous & ~nonfolded).sum()),
             "accepted_cells": int(keep.sum()), "max_uv_edge": max_uv_edge}
    if not keep.any():
        return rgb.new_zeros(10, size, size), rgb.new_zeros(1, size, size), stats
    index = torch.arange(height*width, device=rgb.device).reshape(height, width)
    tl = index[:-1, :-1][keep]
    triangles = torch.cat((torch.stack((tl, tl+1, tl+width), -1),
                           torch.stack((tl+1, tl+width+1, tl+width), -1))).int().contiguous()
    # CUDA rasterizer는 back face를 제거한다. Image row 방향/UV chart 반전과
    # 무관하게 UV 공간의 triangle을 CCW로 맞추되, source vertex 대응은 유지한다.
    clockwise = (area0[keep] < 0).repeat(2)
    triangles = torch.where(clockwise[:, None], triangles[:, [0, 2, 1]], triangles).contiguous()
    coordinates = torch.where(point_valid[None], uv.float(), 0.).reshape(2, -1).T
    if confidence is None:
        z = torch.zeros(height*width, device=rgb.device)
    else:
        z = 1-2*torch.nan_to_num(confidence.float()).clamp(.001, .999).reshape(-1)
    clip = torch.cat((coordinates*2-1, z[:, None], torch.ones_like(z[:, None])), -1)[None].contiguous()
    # nvdiffrast는 아래->위 row, torch 이미지는 위->아래 row이므로 마지막에 flip.
    raster, _ = dr.rasterize(context, clip, triangles, [size, size])
    y, x = torch.meshgrid(torch.arange(height, device=rgb.device), torch.arange(width, device=rgb.device), indexing="ij")
    image_grid = torch.stack((2*(x+.5)/width-1, 2*(y+.5)/height-1), -1).float()
    grid, _ = dr.interpolate(image_grid.reshape(1, -1, 2).contiguous(), raster, triangles)
    grid = grid.flip(1).contiguous()
    # Rasterizer의 subpixel snap으로 원래 triangle 밖의 center가 선택될 수 있다.
    # 실제 barycentric 내부 fragment만 관측으로 인정한다.
    bary = raster[0, ..., :2]
    interior = (bary >= -1e-6).all(-1) & (bary.sum(-1) <= 1+1e-6)
    coverage = ((raster[0, ..., 3] > 0) & interior).flip(0)[None].float()
    safe_features = torch.where(point_valid[None], torch.nan_to_num(features), 0.)
    with torch.autocast(rgb.device.type, enabled=False):
        sampled = F.grid_sample(safe_features[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
        sampled = sampled*coverage
        normal_uv = F.normalize(sampled[3:6], dim=0, eps=1e-6)*coverage
    partial = torch.cat((sampled[:3], coverage, normal_uv, sampled[6:9]), 0)
    return partial, coverage, stats


def repair_source_mask(alpha, uv, probability, xyz, normal, threshold):
    """추론의 foreground 판정. GT valid/UV/topology를 참조하지 않는다."""
    return ((probability >= threshold) & (alpha > .5) & torch.isfinite(probability)
            & torch.isfinite(uv).all(0, keepdim=True) & (uv >= 0).all(0, keepdim=True)
            & (uv <= 1).all(0, keepdim=True) & torch.isfinite(xyz).all(0, keepdim=True)
            & torch.isfinite(normal).all(0, keepdim=True))


def cached_attributes(data, row, crop, device):
    """완료한 XYZ/normal 예측을 읽고 현재 RGB crop과의 좌표 대응을 확인한다."""
    result, metadata = {}, {}
    for task in ("xyz", "normal"):
        path = data.path/"predictions"/task/f"{Path(row['file']).stem}.pt"
        if not path.is_file() or not path.with_suffix(".json").is_file():
            raise FileNotFoundError(f"{task} 예측/metadata가 필요하다: {path}")
        meta = json.loads(path.with_suffix(".json").read_text())
        value = L00.load_tensor(path, device)
        if meta["input_transform"] != INPUT_TRANSFORM or meta["crop"] != crop or value.shape != (3, *crop["source_hw"]):
            raise ValueError(f"{task} 예측의 crop/shape 불일치. 같은 RGB/4:3 crop에서 export하세요.")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{task} cache에 NaN/Inf가 있다.")
        y, x, h, w = (crop[key] for key in ("top", "left", "height", "width"))
        result[task] = value[:, y:y+h, x:x+w]
        metadata[task] = meta
    return result, metadata


def validity_loss(logits, target, dice_weight):
    """양성/음성 pixel 평균을 각각 내어 배경 비율에 따른 BCE 편향을 줄인다."""
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    positive, negative = target > .5, target <= .5
    terms = [bce[mask].mean() for mask in (positive, negative) if mask.any()]
    balanced = torch.stack(terms).mean()
    probability = logits.sigmoid()
    dice = 1-(2*(probability*target).sum()+1)/(probability.sum()+target.sum()+1)
    return balanced+dice_weight*dice


def repair_losses(uv, logits, sample, settings):
    # UV는 기존 foreground L1, mask는 배경을 포함한 이미지 전체를 감독한다.
    mask = sample["labels"]["valid"][None]
    if not mask.any():
        raise ValueError("추가 학습 crop에 UV foreground가 없다.")
    coordinate = (uv-sample["labels"]["uv"][None]).abs().mean(1, keepdim=True)[mask].mean()
    foreground = validity_loss(logits, mask, settings["dice_weight"])
    return coordinate+settings["validity_weight"]*foreground, coordinate, foreground


def save_repair_panel(path, sample, uv, probability, partial, gt_partial, step, metrics, settings):
    valid = sample["labels"]["valid"]
    error = (uv-sample["labels"]["uv"]).norm(dim=0, keepdim=True)*settings["uv_size"]
    images = [sample["rgb"], valid.float().expand(3, -1, -1), probability.expand(3, -1, -1),
              gt_partial[:3], partial[:3], partial[3:4].expand(3, -1, -1),
              heatmap(error, valid, 32), (gt_partial[4:7]*.5+.5)*gt_partial[3:4],
              (partial[4:7]*.5+.5)*partial[3:4]]
    titles = ["Input (fixed center crop)", "GT foreground: metrics only", "Predicted foreground probability",
              "GT UV resampling: diagnostic only", "Actual predicted partial UV RGB", "Actual predicted UV visibility",
              "UV error: blue=0, red>=32 texels", "GT UV normal", "Actual predicted UV normal"]
    width, height, header = 300, 400, 64
    canvas = Image.new("RGB", (width*3, header+3*(height+28)), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), f"UV repair | optimizer step {step} | FP32 decoder | predicted mask >= {settings['validity_threshold']}", fill="white")
    draw.text((10, 27), " | ".join(f"{k}={v:.4g}" for k, v in metrics.items() if k in ("uv_error_texels", "validity_iou", "source_precision")), fill="white")
    draw.text((10, 46), "Actual predictions use learned foreground + cached XYZ/normal. GT is comparison only.", fill="white")
    for index, (value, title) in enumerate(zip(images, titles)):
        array = (value.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()*255).round().astype(np.uint8)
        tile = Image.fromarray(array)
        scale = min(width/tile.width, height/tile.height)
        tile = tile.resize((round(tile.width*scale), round(tile.height*scale)), Image.Resampling.BILINEAR)
        x, y = (index % 3)*width, header+(index//3)*(height+28)
        draw.text((x+5, y+6), title, fill="white")
        canvas.paste(tile, (x+(width-tile.width)//2, y+28+(height-tile.height)//2))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=95)


@torch.no_grad()
def validate_uv_repair(model, data, device, out, step, rows, settings, context):
    """최종 partial UV까지 검증한다. GT mask는 loss/지표/GT 비교 패널에만 쓴다."""
    if not rows:
        raise ValueError("validation split이 필요하다.")
    was_training = model.training
    model.eval()
    pixels, uv_error, uv_l1, mask_loss_sum = 0, 0., 0., 0.
    tp, fp, fn, source_tp, source_fp, source_fn = 0, 0, 0, 0, 0, 0
    support, gt_support, photo_sum, overlap_pixels = 0, 0, 0., 0
    preview = None
    settings = dict(settings, uv_size=data.manifest["config"]["uv_size"])
    try:
        for row in rows:
            sample = prepare_input(data.sample(row, device), model.input_size)
            attributes, _ = cached_attributes(data, row, sample["crop"], device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                uv, logit = model.predict(sample["rgb"][None])
            if not torch.isfinite(uv).all() or not torch.isfinite(logit).all():
                raise FloatingPointError("UV/validity prediction NaN/Inf")
            uv, probability = uv[0], logit[0].sigmoid()
            target = sample["labels"]["valid"]
            predicted = probability >= settings["validity_threshold"]
            source = repair_source_mask(sample["alpha"], uv, probability, attributes["xyz"], attributes["normal"], settings["validity_threshold"])
            tp += int((predicted & target).sum()); fp += int((predicted & ~target).sum()); fn += int((~predicted & target).sum())
            source_tp += int((source & target).sum()); source_fp += int((source & ~target).sum()); source_fn += int((~source & target).sum())
            count = int(target.sum())
            if not count:
                raise ValueError(f"validation foreground 없음: {row['file']}")
            difference = uv-sample["labels"]["uv"]
            uv_error += float(difference.norm(dim=0, keepdim=True)[target].sum())*settings["uv_size"]
            uv_l1 += float(difference.abs().mean(0, keepdim=True)[target].sum())
            pixels += count
            mask_loss_sum += float(validity_loss(logit, target[None], settings["dice_weight"]))
            partial, _, stats = resample_partial_uv(sample["rgb"], uv, attributes["xyz"], attributes["normal"], source,
                              settings["uv_size"], context, probability, settings["max_uv_edge"])
            labels = sample["labels"]
            gt, _, _ = resample_partial_uv(sample["rgb"], labels["uv"], labels["xyz"], labels["normal"], target,
                              settings["uv_size"], context, max_uv_edge=settings["max_uv_edge"])
            support += int(partial[3].sum()); gt_support += int(gt[3].sum())
            overlap = (partial[3:4] > .5) & (gt[3:4] > .5)
            overlap_pixels += int(overlap.sum())
            photo_sum += float((partial[:3]-gt[:3]).abs().mean(0, keepdim=True)[overlap].sum())
            if preview is None:
                preview = (sample, uv, probability, partial, gt, row, stats)
        metrics = {"uv_l1": uv_l1/pixels, "uv_error_texels": uv_error/pixels,
                   "validity_loss": mask_loss_sum/len(rows), "validity_iou": tp/max(1, tp+fp+fn),
                   "validity_precision": tp/max(1, tp+fp), "validity_recall": tp/max(1, tp+fn),
                   "source_precision": source_tp/max(1, source_tp+source_fp),
                   "source_recall": source_tp/max(1, source_tp+source_fn),
                   "outside_source_fraction": source_fp/max(1, source_tp+source_fp),
                   "predicted_uv_texels": support, "gt_uv_texels": gt_support,
                   "partial_rgb_l1_overlap": photo_sum/overlap_pixels if overlap_pixels else None}
        directory = out/"validation"
        save_repair_panel(directory/f"{step}.jpg", *preview[:5], step, metrics, settings)
        L00.append_log(directory/"metrics.jsonl", {"step": step, **metrics,
                       "frames": [r["file"] for r in rows], "preview_frame": preview[5]["file"],
                       "preview_sampling": preview[6], "variant": REPAIR_VARIANT,
                       "input_transform": INPUT_TRANSFORM, "scope": data.manifest["scope"]})
        return metrics
    finally:
        model.train(was_training)


@torch.no_grad()
def export_uv_repair(model, data, device, run_name, step, settings):
    """기존 uv/xyz/normal cache는 유지하고 새 UV/probability를 run 폴더에 저장한다."""
    if not isinstance(model, UVValidityPredictor):
        raise ValueError("UV validity checkpoint가 필요하다.")
    model.eval()
    fingerprint = L00.sha256(data.path/"manifest.json")
    for row in tqdm(data.manifest["samples"], desc="01 UV repair predictions", dynamic_ncols=True, mininterval=1):
        original = data.sample(row, device)
        sample = prepare_input(original, model.input_size)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            uv, logits = model.predict(sample["rgb"][None])
        if not torch.isfinite(uv).all() or not torch.isfinite(logits).all():
            raise FloatingPointError("export UV/validity NaN/Inf")
        crop = sample["crop"]
        y, x, h, w = (crop[key] for key in ("top", "left", "height", "width"))
        full_uv = uv.new_full((2, *crop["source_hw"]), -1)
        full_probability = uv.new_zeros((1, *crop["source_hw"]))
        full_uv[:, y:y+h, x:x+w] = uv[0]
        full_probability[:, y:y+h, x:x+w] = logits[0].sigmoid()
        for task, tensor in (("uv", full_uv), ("validity", full_probability)):
            path = data.path/"predictions"/run_name/task/f"{Path(row['file']).stem}.pt"
            L00.save_tensor(path, tensor.cpu())
            L00.save_json(path.with_suffix(".json"), {"run": run_name, "step": step, "crop": crop,
                "input_transform": INPUT_TRANSFORM, "variant": REPAIR_VARIANT, "repair_settings": settings,
                "data_sha256": fingerprint})


@torch.no_grad()
def compose_uv_repair(data, device, run_name, settings=None):
    """GT 없는 최종 추론. 02가 읽는 partial_pred를 이 run의 예측으로 갱신한다."""
    import nvdiffrast.torch as dr
    context = dr.RasterizeCudaContext(device=device)
    # 모든 frame의 파일/좌표계를 먼저 확인한다. 빠진 예측을 GT나 사람 mask로 대신하지 않는다.
    records = []
    fingerprint = L00.sha256(data.path/"manifest.json")
    first_identity = None
    for row in data.manifest["samples"]:
        stem = Path(row["file"]).stem
        metadata = {}
        for task in ("uv", "validity"):
            path = data.path/"predictions"/run_name/task/f"{stem}.pt"
            if not path.is_file() or not path.with_suffix(".json").is_file():
                raise FileNotFoundError(f"추가 학습 예측이 필요하다: {path}. --stage finetune-uv 또는 export를 실행하세요.")
            metadata[task] = json.loads(path.with_suffix(".json").read_text())
        if metadata["uv"] != metadata["validity"] or metadata["uv"]["variant"] != REPAIR_VARIANT:
            raise ValueError("UV/validity의 run/step/설정 불일치. 다시 export하세요.")
        identity = (metadata["uv"]["run"], metadata["uv"]["step"], metadata["uv"]["repair_settings"])
        if metadata["uv"].get("data_sha256") != fingerprint or identity[0] != run_name:
            raise ValueError("repair cache의 dataset/run 불일치. 다시 export하세요.")
        if first_identity is None:
            first_identity = identity
        elif identity != first_identity:
            raise ValueError("frame마다 다른 repair step/설정이 섞여 있다. 전체 frame을 다시 export하세요.")
        crop = metadata["uv"]["crop"]
        _, attributes_metadata = cached_attributes(data, row, crop, "cpu")
        records.append((row, metadata["uv"], attributes_metadata))
    for row, metadata, attributes_metadata in tqdm(records, desc="01 partial UV resampling", dynamic_ncols=True, mininterval=1):
        chosen = dict(metadata["repair_settings"], **(settings or {}))
        stem = Path(row["file"]).stem
        sample = data.sample(row, device)
        uv = L00.load_tensor(data.path/"predictions"/run_name/"uv"/f"{stem}.pt", device)
        probability = L00.load_tensor(data.path/"predictions"/run_name/"validity"/f"{stem}.pt", device)
        xyz = L00.load_tensor(data.path/"predictions/xyz"/f"{stem}.pt", device)
        normal = L00.load_tensor(data.path/"predictions/normal"/f"{stem}.pt", device)
        if uv.shape != (2, *sample["rgb"].shape[-2:]) or probability.shape != (1, *sample["rgb"].shape[-2:]):
            raise ValueError("repair cache의 shape 불일치")
        valid = repair_source_mask(sample["alpha"], uv, probability, xyz, normal, chosen["validity_threshold"])
        partial, coverage, stats = resample_partial_uv(sample["rgb"], uv, xyz, normal, valid,
                           data.manifest["config"]["uv_size"], context, probability, chosen["max_uv_edge"])
        L00.save_tensor(data.path/"partial_pred"/f"{stem}.pt", {"partial": partial.cpu(), "weight": coverage.cpu(),
            "weight_kind": "raster_coverage", "origin": REPAIR_VARIANT, "source_frame": row["file"],
            "uv_predictor": metadata, "attribute_predictors": attributes_metadata, "sampling": stats,
            "validity_threshold": chosen["validity_threshold"], "gt_used_for_prediction": False})
        L00.save_panels(L00.work_root(data.manifest["config"])/"outputs/01"/run_name/"predictions"/f"{stem}.jpg",
            [sample["rgb"], partial[:3], partial[3:4], (partial[4:7]*.5+.5)*partial[3:4]],
            ["Input", "Repaired partial UV RGB", "Predicted visibility", "Repaired UV normal"])


def train_uv_repair(config, data, device, steps, init_from=None, resume=False,
                    run_name="uv_repair", val_every=100, val_frames=8, save=True):
    """기존 UV weights -> 새 validity head warmup -> UV/backbone 공동 추가 학습.

    Step은 추가 학습의 횟수다. 기존 50K-step optimizer는 새 목표에 맞게 초기화한다.
    같은 repair run의 --resume은 optimizer/RNG/step을 복원한다.
    """
    import nvdiffrast.torch as dr
    settings = config["uv_repair"]
    if steps < 1 or val_every < 1 or val_frames < 0 or settings["warmup_steps"] < 0:
        raise ValueError("steps/val_every>=1, val_frames/warmup_steps>=0이 필요하다.")
    if not 0 < settings["validity_threshold"] < 1 or settings["max_uv_edge"] <= 0:
        raise ValueError("validity_threshold는 (0,1), max_uv_edge는 양수여야 한다.")
    if settings["checkpoint_every"] < 1 or settings["validity_weight"] <= 0 or settings["dice_weight"] < 0:
        raise ValueError("checkpoint_every>=1, validity_weight>0, dice_weight>=0이 필요하다.")
    if any(settings[key] <= 0 for key in ("encoder_lr", "head_lr", "validity_lr")):
        raise ValueError("추가 학습의 learning rate는 양수여야 한다.")
    if resume and init_from:
        raise ValueError("--resume과 --init-from을 함께 사용할 수 없다.")
    out, path = run_paths(config, "uv", run_name)
    if save and not resume and (path.exists() or (out/"training.jsonl").exists()):
        raise FileExistsError(f"기존 run을 보존한다: {out}. --resume 또는 다른 --run-name을 지정하세요.")
    rows = validation_rows(data, val_frames)
    if not rows:
        raise ValueError("validation split이 필요하다.")
    # Before a long run, check every required XYZ/normal export using CPU only.
    input_size = tuple(config["sapiens"]["input_size"])
    for row in data.manifest["samples"]:
        sample = prepare_input(data.sample(row), input_size)
        cached_attributes(data, row, sample["crop"], "cpu")
    source_path = path if resume else init_from or L00.work_root(config)/"checkpoints/01/crop/uv.pt"
    base, source = load_predictor(source_path, "uv", config, data, device)
    if resume:
        if not isinstance(base, UVValidityPredictor):
            raise ValueError("--resume은 repair checkpoint를 사용해야 한다.")
        model = base
    else:
        model = base if isinstance(base, UVValidityPredictor) else UVValidityPredictor(base)
    model.train()
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": settings["encoder_lr"]},
        {"params": model.decode_head.parameters(), "lr": settings["head_lr"]},
        {"params": model.validity_head.parameters(), "lr": settings["validity_lr"]}], weight_decay=.01)
    start = L00.training_start(path, model, optimizer, config, data, True) if resume else 0
    if steps <= start:
        raise ValueError(f"--steps는 최종 repair step이다. 현재 {start}보다 크게 지정하세요.")
    if not resume:
        L00.save_json(out/"run.json", {"variant": REPAIR_VARIANT, "input_transform": INPUT_TRANSFORM,
            "parent": source, "config": config, "code_hashes": L00.code_hashes(), "run": run_name,
            "val_every": val_every, "validation_frames": [r["file"] for r in rows], "optimizer_reset": True,
            "implementation_choice": "Auxiliary validity head and UV triangulation are our reproduction choices, not released FiCA code."})
    extra = {"task": "uv", "variant": REPAIR_VARIANT, "input_transform": INPUT_TRANSFORM,
             "run": run_name, "repair_settings": settings}
    context = dr.RasterizeCudaContext(device=device)
    accumulation = config["training"]["grad_accumulation"]
    model.backbone.requires_grad_(start >= settings["warmup_steps"])
    model.decode_head.requires_grad_(start >= settings["warmup_steps"])
    with tqdm(total=steps, initial=start, desc="01 UV repair", unit="step", dynamic_ncols=True, mininterval=1) as bar:
        recent = validate_uv_repair(model, data, device, out, start, rows, settings, context) if start == 0 or start % val_every == 0 else {}
        for step in range(start, steps):
            warmup = step < settings["warmup_steps"]
            model.backbone.requires_grad_(not warmup)
            model.decode_head.requires_grad_(not warmup)
            optimizer.zero_grad(set_to_none=True)
            total_loss, uv_loss, mask_loss = 0., 0., 0.
            for _ in range(accumulation):
                sample = prepare_input(data.sample(data.training_row(), device), model.input_size,
                                       augment=config["sapiens"]["augmentation"])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    uv, logits = model.predict(sample["rgb"][None])
                    loss, coordinate, foreground = repair_losses(uv, logits, sample, settings)
                (loss/accumulation).backward()
                total_loss += float(loss.detach())/accumulation
                uv_loss += float(coordinate.detach())/accumulation
                mask_loss += float(foreground.detach())/accumulation
            report = L00.gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["clip_grad"])
            optimizer.step()
            L00.append_log(out/"training.jsonl", {"step": step+1, "loss": total_loss, "uv_l1": uv_loss,
                           "validity_loss": mask_loss, "warmup": warmup, "gradient": report})
            if (step+1) % val_every == 0:
                recent = validate_uv_repair(model, data, device, out, step+1, rows, settings, context)
            bar.set_postfix(uv=f"{uv_loss:.5f}", mask=f"{mask_loss:.4f}",
                            iou=f"{recent['validity_iou']:.3f}" if recent else "pending",
                            phase="mask" if warmup else "joint", refresh=False)
            bar.update(1)
            if save and (step+1) % settings["checkpoint_every"] == 0:
                L00.checkpoint(path, model, optimizer, config, data, step+1, extra)
    if save and steps % settings["checkpoint_every"] != 0:
        L00.checkpoint(path, model, optimizer, config, data, steps, extra)
    export_uv_repair(model, data, device, run_name, steps, settings)
    del model, optimizer
    torch.cuda.empty_cache()


def compose_predictions(data, device):
    """3개의 실제 task prediction으로 partial UV를 만든다. GT로 fallback하지 않는다."""
    for row in data.manifest["samples"]:
        sample = data.sample(row, device)
        stem = Path(row["file"]).stem
        values = {task: L00.load_tensor(data.path / "predictions" / task / f"{stem}.pt", device) for task in ("uv", "xyz", "normal")}
        metadata = {}
        for task in values:
            path = data.path / "predictions" / task / f"{stem}.json"
            if path.exists():
                metadata[task] = json.loads(path.read_text())
            if values[task].shape != ((2 if task == "uv" else 3), *sample["rgb"].shape[-2:]):
                raise ValueError(f"{task}: prediction과 원본 RGB 해상도가 다르다.")
        if metadata:
            if len(metadata) != 3 or any((m["crop"], m["input_transform"]) != (metadata["uv"]["crop"], metadata["uv"]["input_transform"]) for m in metadata.values()):
                raise ValueError("세 task의 crop 전처리가 다르다. UV/XYZ/normal을 모두 새 입력 방식으로 예측하세요.")
        # 비교용 이전 규칙. FiCA의 visibility 세부 구현은 공개되지 않았다.
        valid = (sample["alpha"] > .5) & torch.isfinite(values["xyz"]).all(0, keepdim=True)
        partial, weight = L00.splat_partial_uv(sample["rgb"], values["uv"], values["xyz"], values["normal"], valid, data.manifest["config"]["uv_size"])
        L00.save_tensor(data.path / "partial_pred" / f"{stem}.pt", {"partial": partial.cpu(), "weight": weight.cpu(), "origin": "trained_sapiens_predictions", "source_frame": row["file"], "predictors": metadata})
        L00.save_panels(L00.work_root(data.manifest["config"]) / "outputs/01" / f"{stem}.jpg", [sample["rgb"], sample["partial_gt"][:3], partial[:3], partial[3:4], partial[4:7]*.5+.5],
                        ["Input", "GT-correspondence UV", "Predicted UV RGB", "Predicted visibility", "Predicted UV normal"])


def main():
    parser = L00.arguments(__doc__)
    parser.add_argument("--stage", choices=("train", "finetune-uv", "validate", "export", "predict"), default="train")
    parser.add_argument("--task", choices=("uv", "xyz", "normal"), default="uv")
    parser.add_argument("--run-name", help="기본: train/validate=crop, finetune-uv/export/predict=uv_repair")
    parser.add_argument("--val-every", type=int, default=100, help="Optimizer step 기준 validation 간격 (기본 100)")
    parser.add_argument("--val-frames", type=int, default=8, help="고정 held-out frame 개수; 0은 전체")
    parser.add_argument("--checkpoint", type=Path, help="validate/export에서 읽을 checkpoint; 기본은 현재 run")
    parser.add_argument("--init-from", type=Path, help="기존 weight로 새 학습 시작. Optimizer/step은 초기화한다.")
    parser.add_argument("--warmup-steps", type=int, help="추가 학습 초기에 validity head만 학습할 step (기본 100)")
    parser.add_argument("--validity-threshold", type=float, help="추가 학습/추론의 foreground 확률 문턱값 (기본 .7)")
    parser.add_argument("--legacy-splat", action="store_true", help="predict에서 이전 person-mask/point-splat 방식과 비교")
    args = parser.parse_args()
    if args.val_every < 1 or args.val_frames < 0:
        parser.error("--val-every >= 1, --val-frames >= 0 이어야 한다.")
    if args.resume and (args.stage not in ("train", "finetune-uv") or args.init_from):
        parser.error("--resume은 train/finetune-uv에서만 사용하고 --init-from과 함께 쓰지 않는다.")
    if args.checkpoint and args.stage not in ("validate", "export"):
        parser.error("--checkpoint는 validate/export에서 사용한다. 학습 초기화는 --init-from이다.")
    if args.init_from and args.stage not in ("train", "finetune-uv"):
        parser.error("--init-from은 train/finetune-uv에서 사용한다.")
    if args.stage == "finetune-uv" and args.task != "uv":
        parser.error("finetune-uv는 --task uv만 지원한다. XYZ/normal은 기존 예측을 재사용한다.")
    if args.legacy_splat and args.stage != "predict":
        parser.error("--legacy-splat은 predict 전용이다.")
    config = L00.read_config(args.config, args.smoke)
    if args.warmup_steps is not None:
        if args.stage != "finetune-uv" or args.warmup_steps < 0:
            parser.error("--warmup-steps는 finetune-uv에서 0 이상으로 지정한다.")
        config["uv_repair"]["warmup_steps"] = args.warmup_steps
    if args.validity_threshold is not None:
        if not 0 < args.validity_threshold < 1:
            parser.error("--validity-threshold는 0과 1 사이여야 한다.")
        config["uv_repair"]["validity_threshold"] = args.validity_threshold
    repair_stage = args.stage in ("finetune-uv", "predict") or (args.stage == "export" and args.task == "uv")
    run_name = args.run_name or ("uv_repair" if repair_stage else "crop")
    device = L00.setup_device(args.device, config["seed"])
    data = L00.PreparedData(args.data)
    if args.stage == "train":
        train_task(config, data, args.task, device, args.steps or config["training"]["steps"],
                   resume=args.resume, val_every=args.val_every, val_frames=args.val_frames,
                   run_name=run_name, init_from=args.init_from)
    elif args.stage == "finetune-uv":
        train_uv_repair(config, data, device, args.steps or config["uv_repair"]["steps"],
                        init_from=args.init_from, resume=args.resume, run_name=run_name,
                        val_every=args.val_every, val_frames=args.val_frames)
    elif args.stage in ("validate", "export"):
        out, path = run_paths(config, args.task, run_name)
        path = args.checkpoint or path
        model, source = load_predictor(path, args.task, config, data, device)
        if args.run_name is None:
            run_name = source["extra"].get("run", "uv_repair" if isinstance(model, UVValidityPredictor) else "crop")
            out, _ = run_paths(config, args.task, run_name)
        if isinstance(model, UVValidityPredictor):
            settings = dict(source["extra"]["repair_settings"])
            if args.validity_threshold is not None:
                settings["validity_threshold"] = args.validity_threshold
            if args.stage == "export":
                export_uv_repair(model, data, device, run_name, source["source_step"], settings)
            else:
                import nvdiffrast.torch as dr
                context = dr.RasterizeCudaContext(device=device)
                L00.save_json(out/"validation"/"source.json", source)
                metrics = validate_uv_repair(model, data, device, out, source["source_step"],
                            validation_rows(data, args.val_frames), settings, context)
                print(f"01 UV repair validation: {metrics} | {out / 'validation'}")
            return
        mean = torch.tensor(data.manifest["xyz_mean"], device=device)[None, :, None, None]
        std = torch.tensor(data.manifest["xyz_std"], device=device)[None, :, None, None]
        if args.stage == "export":
            export_predictions(model, data, device, mean, std, run_name, source["source_step"])
        else:
            L00.save_json(out/"validation"/"source.json", dict(source, input_transform=INPUT_TRANSFORM))
            metrics = validate(model, data, device, out, source["source_step"], mean, std, validation_rows(data, args.val_frames))
            print(f"01 {args.task} validation: {metrics} | {out / 'validation'}")
    else:
        if args.legacy_splat:
            compose_predictions(data, device)
        else:
            overrides = {"validity_threshold": args.validity_threshold} if args.validity_threshold is not None else None
            compose_uv_repair(data, device, run_name, overrides)


if __name__ == "__main__":
    main()
