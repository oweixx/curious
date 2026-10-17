"""01 — 한 장의 RGB에서 UV / canonical XYZ / normal을 예측하고 partial UV를 만든다.

세 task는 독립적인 Sapiens-1B encoder를 fine-tune한다(FiCA C.1).
UV/XYZ: pre-trained encoder + depth-style decoder. Normal: official normal checkpoint.
큰 OpenMMLab framework 대신 같은 state_dict 이름/연산의 PyTorch port를 사용한다.
Encoder/head를 더미 CNN으로 대체하지 않는다. 모델 크기는 smoke에서도 실제 1B다.

CUDA_VISIBLE_DEVICES=0 python 01_partial_uv.py --stage train --task uv
CUDA_VISIBLE_DEVICES=0 python 01_partial_uv.py --stage train --task xyz
CUDA_VISIBLE_DEVICES=0 python 01_partial_uv.py --stage train --task normal
CUDA_VISIBLE_DEVICES=0 python 01_partial_uv.py --stage predict
"""

import importlib.util
from pathlib import Path
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as recompute

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

    def forward(self, features):
        return self.conv_seg(self.conv_layers(self.deconv_layers(features)))


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


def augment_sample(sample):
    """모든 dense label에 같은 crop/scale을 적용하고 RGB만 photometric 변형한다."""
    height,width=sample["rgb"].shape[-2:]
    ratio=float(torch.empty(()).uniform_(.8,1.))
    h,w=max(16,round(height*ratio)),max(16,round(width*ratio))
    top=int(torch.randint(height-h+1,()))
    left=int(torch.randint(width-w+1,()))
    def crop(value,nearest=False):
        value=value[:,top:top+h,left:left+w][None].float()
        return F.interpolate(value,size=(height,width),mode="nearest" if nearest else "bilinear",**({} if nearest else {"align_corners":False}))[0]
    labels={key:crop(value,key=="valid") for key,value in sample["labels"].items()}
    labels["valid"]=labels["valid"]>.5
    for key in ("normal","normal_world"):
        labels[key]=F.normalize(labels[key],dim=0)
    rgb=crop(sample["rgb"])
    contrast=float(torch.empty(()).uniform_(.9,1.1))
    brightness=float(torch.empty(()).uniform_(-.05,.05))
    rgb=((rgb-rgb.mean((-2,-1),keepdim=True))*contrast+rgb.mean((-2,-1),keepdim=True)+brightness).clamp(0,1)
    return dict(sample,rgb=rgb,labels=labels)


def train_task(config, data, task, device, steps, save=True, resume=False):
    model = DensePredictor(task, device, config["sapiens"]["input_size"]).train()
    optimizer = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": config["sapiens"]["encoder_lr"]},
                                  {"params": model.decode_head.parameters(), "lr": config["sapiens"]["head_lr"]}], weight_decay=.01)
    out = L00.work_root(config) / "outputs/01" / task
    checkpoint_path=L00.work_root(config)/"checkpoints/01"/f"{task}.pt"
    start=L00.training_start(checkpoint_path,model,optimizer,config,data,resume) if save else 0
    mean = torch.tensor(data.manifest["xyz_mean"], device=device)[None, :, None, None]
    std = torch.tensor(data.manifest["xyz_std"], device=device)[None, :, None, None]
    rows = data.samples("train")
    accumulation=config["training"]["grad_accumulation"]
    for step in range(start,steps):
        optimizer.zero_grad(set_to_none=True)
        mean_loss=0.
        for micro in range(accumulation):
            sample=data.sample(data.training_row(),device)
            if config["sapiens"]["augmentation"]:
                sample=augment_sample(sample)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted = model(sample["rgb"][None])
                loss = prediction_loss(task, predicted, sample, mean, std)
            (loss/accumulation).backward()
            mean_loss+=float(loss.detach())/accumulation
        report = L00.gradients(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["clip_grad"])
        optimizer.step()
        L00.append_log(out / "training.jsonl", {"step": step+1, "loss": mean_loss, "gradient": report, "scope": data.manifest["scope"]})
        print(f"01 {task} step {step+1}/{steps}: {mean_loss:.6f}", flush=True)
        if save and (step+1)%config["training"]["checkpoint_every"]==0:
            L00.checkpoint(checkpoint_path,model,optimizer,config,data,step+1,{"task":task})
    if save:
        L00.checkpoint(L00.work_root(config) / "checkpoints/01" / f"{task}.pt", model, optimizer, config, data, steps, {"task": task})
    # 이후 단계는 cache를 읽는다. GT/예측을 파일 이름과 metadata로 분리한다.
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in data.manifest["samples"]:
            sample = data.sample(row, device)
            predicted = model(sample["rgb"][None])
            if task == "xyz":
                predicted = predicted * std + mean
            stem = Path(row["file"]).stem
            L00.save_tensor(data.path / "predictions" / task / f"{stem}.pt", predicted[0].cpu())
    del model, optimizer
    torch.cuda.empty_cache()


def compose_predictions(data, device):
    """3개의 실제 task prediction으로 partial UV를 만든다. GT로 fallback하지 않는다."""
    for row in data.manifest["samples"]:
        sample = data.sample(row, device)
        stem = Path(row["file"]).stem
        values = {task: L00.load_tensor(data.path / "predictions" / task / f"{stem}.pt", device) for task in ("uv", "xyz", "normal")}
        # FiCA의 비공개 visibility estimator를 대신하는 명시적 foreground rule.
        valid = (sample["alpha"] > .5) & torch.isfinite(values["xyz"]).all(0, keepdim=True)
        partial, weight = L00.splat_partial_uv(sample["rgb"], values["uv"], values["xyz"], values["normal"], valid, data.manifest["config"]["uv_size"])
        L00.save_tensor(data.path / "partial_pred" / f"{stem}.pt", {"partial": partial.cpu(), "weight": weight.cpu(), "origin": "trained_sapiens_predictions", "source_frame": row["file"]})
        L00.save_panels(L00.work_root(data.manifest["config"]) / "outputs/01" / f"{stem}.jpg", [sample["rgb"], sample["partial_gt"][:3], partial[:3], partial[3:4], partial[4:7]*.5+.5],
                        ["Input", "GT-correspondence UV", "Predicted UV RGB", "Predicted visibility", "Predicted UV normal"])


def main():
    parser = L00.arguments(__doc__)
    parser.add_argument("--stage", choices=("train", "predict"), default="train")
    parser.add_argument("--task", choices=("uv", "xyz", "normal"), default="uv")
    args = parser.parse_args()
    config = L00.read_config(args.config, args.smoke)
    device = L00.setup_device(args.device, config["seed"])
    data = L00.PreparedData(args.data)
    if args.stage == "train":
        train_task(config, data, args.task, device, args.steps or config["training"]["steps"],resume=args.resume)
    else:
        compose_predictions(data, device)


if __name__ == "__main__":
    main()
