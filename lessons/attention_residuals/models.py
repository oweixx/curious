"""Attention Residuals의 수식을 직접 옮긴 작은 PyTorch 구현.

읽는 순서: TokenAttention -> DepthAttention -> RecallTransformer.forward.
출처: arxiv.org/abs/2603.15031, Eq.1--6 / Algorithm 1.
공식 README의 의사코드와 수식을 참고한 독립 구현이며 공식 학습 코드가 아니다.

중요한 축: token attention은 T축, depth attention은 S(이전 출력/블록)축.
논문의 layer 수 L은 ATTN/MLP 각각을 센다. 이 코드의 depth는 Transformer 쌍 수.
따라서 L=2*depth, block_size=4는 ATTN,MLP,ATTN,MLP를 한 블록에 담는다.
"""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # 평균을 빼는 LayerNorm과 구분한다. 이 lesson은 float32/float64만 사용.
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


class TokenAttention(nn.Module):
    """일반적인 causal multi-head self-attention. [B,T,D] -> [B,T,D]."""
    def __init__(self, dim, heads):
        super().__init__()
        if dim % heads:
            raise ValueError("dim은 heads의 배수여야 합니다.")
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d // self.heads)
        mask = torch.ones(t, t, device=x.device, dtype=torch.bool).tril()
        weights = scores.masked_fill(~mask, -torch.inf).softmax(dim=-1)
        mixed = (weights @ v).transpose(1, 2).reshape(b, t, d)
        return self.proj(mixed), weights


class DepthAttention(nn.Module):
    """논문의 AttnRes 연산. source마다 [B,T,D], stack 후 [S,B,T,D].

    query는 현재 hidden state의 projection이 아니라 layer별 Parameter[D]다.
    key 내용이 입력마다 달라지므로 alpha[S,B,T]는 input-dependent다.
    key만 RMSNorm하고 value는 원래 크기를 유지한다. sqrt(D)로 나누지 않는다.
    static은 input-independent mixing을 비교하기 위한 선택적 ablation이다.
    """
    def __init__(self, dim, key_norm=True, static_sources=None):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(dim))  # 논문 §5: 반드시 zero-init.
        self.norm = RMSNorm(dim) if key_norm else nn.Identity()
        self.static_logits = (
            nn.Parameter(torch.zeros(static_sources)) if static_sources else None
        )
        if static_sources:
            self.query.requires_grad_(False)
            self.norm.requires_grad_(False)

    def scores(self, values):
        if self.static_logits is not None:
            return self.static_logits[:len(values), None, None].expand(values.shape[:-1])
        return torch.einsum("d,sbtd->sbt", self.query, self.norm(values))

    def forward(self, sources):
        values = torch.stack(sources)
        weights = self.scores(values).softmax(dim=0)  # token축이 아닌 source/depth축!
        return torch.einsum("sbt,sbtd->btd", weights, values), weights


def prepare_inter_block(mixers, completed):
    """Algorithm 1의 Phase 1: 이미 완성된 block에 대한 여러 query를 함께 계산.

    mixer마다 key RMSNorm의 affine weight가 다르므로 query에 흡수한다.
    q_eff = q * norm.weight; RMS denominator는 모든 mixer에서 공유할 수 있다.
    반환: 각 query의 정규화된 output과 logsumexp. 학습 graph도 유지된다.
    """
    values = torch.stack(completed)
    if not all(isinstance(m.norm, RMSNorm) and m.static_logits is None for m in mixers):
        raise ValueError("two-phase 예제는 논문 기본형 RMSNorm + pseudo-query에 적용합니다.")
    if len({m.norm.eps for m in mixers}) != 1:
        raise ValueError("batched Phase 1에는 같은 RMSNorm eps가 필요합니다.")
    keys = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + mixers[0].norm.eps)
    queries = torch.stack([m.query * m.norm.weight for m in mixers])
    scores = torch.einsum("rd,sbtd->rsbt", queries, keys)
    weights = scores.softmax(dim=1)
    return torch.einsum("rsbt,sbtd->rbtd", weights, values), scores.logsumexp(dim=1)


def merge_partial(mixer, inter_output, inter_lse, partial):
    """Phase 2: 과거 block 결과와 현재 partial sum을 정확한 softmax 비율로 병합.

    두 attention output을 그냥 더하면 틀리다. 분모(partition function)를 합쳐야 한다.
    logaddexp를 써서 큰 logits에서도 exp overflow를 피한다.
    """
    if partial is None:
        return inter_output
    partial_score = mixer.scores(partial.unsqueeze(0))[0]
    total_lse = torch.logaddexp(inter_lse, partial_score)
    inter_weight = (inter_lse - total_lse).exp()[..., None]
    partial_weight = (partial_score - total_lse).exp()[..., None]
    return inter_weight * inter_output + partial_weight * partial


class Sublayer(nn.Module):
    def __init__(self, dim, heads, is_attention):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.is_attention = is_attention
        if is_attention:
            self.fn = TokenAttention(dim, heads)
        else:
            self.fn = nn.Sequential(
                nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
            )

    def forward(self, x):
        if self.is_attention:
            return self.fn(self.norm(x))
        return self.fn(self.norm(x)), None


@dataclass
class Config:
    variant: str = "full"
    depth: int = 6
    dim: int = 64
    heads: int = 4
    block_size: int = 4
    pairs: int = 8


class RecallTransformer(nn.Module):
    """token attention/MLP/입출력을 공유하는 baseline / Full / Block 비교 모델.

    인공 key-color recall 과제이며 Kimi Linear/MoE의 재현은 아니다.
    AttnRes 연산은 논문의 수식을 따른다. train/inference의 forward는 같다.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config
        if c.variant not in ("baseline", "full", "block", "static", "no_key_norm"):
            raise ValueError("지원하지 않는 variant")
        if min(c.depth, c.dim, c.heads, c.block_size, c.pairs) < 1 or c.pairs > 26:
            raise ValueError("양수 설정이 필요하며 pairs는 26 이하여야 합니다.")
        self.key_embed = nn.Embedding(c.pairs, c.dim)
        self.color_embed = nn.Embedding(9, c.dim)
        self.position = nn.Parameter(torch.randn(1, c.pairs + 1, c.dim) * .02)
        self.layers = nn.ModuleList(
            [Sublayer(c.dim, c.heads, i % 2 == 0) for i in range(2 * c.depth)]
        )
        self.final_norm = RMSNorm(c.dim)
        self.classifier = nn.Linear(c.dim, 8)
        # 마지막 출력 head 앞에도 별도의 AttnRes가 있다 (논문 Eq.6 뒤 문단).
        self.mixers = nn.ModuleList()
        if c.variant != "baseline":
            for _ in range(2 * c.depth + 1):
                mixer = DepthAttention(
                    c.dim,
                    key_norm=c.variant != "no_key_norm",
                    static_sources=2 * c.depth + 1 if c.variant == "static" else None,
                )
                self.mixers.append(mixer)

    def forward(self, keys, colors, capture=False, two_phase=False):
        c = self.config
        if two_phase and c.variant != "block":
            raise ValueError("two_phase=True는 block variant에서만 사용합니다.")
        embedding = self.key_embed(keys) + self.color_embed(colors) + self.position
        hidden = embedding
        completed, partial = [embedding], None
        trace = {"depth_weights": [], "token_weights": [], "inputs": [], "outputs": []}
        block_size = c.block_size if c.variant == "block" else 1
        inter_outputs = inter_lses = None
        for i, layer in enumerate(self.layers):
            weights = None
            if c.variant != "baseline":
                sources = completed + ([] if partial is None else [partial])
                if two_phase:
                    if i % block_size == 0:
                        end = min(i + block_size, len(self.layers))
                        inter_outputs, inter_lses = prepare_inter_block(
                            self.mixers[i:end], completed
                        )
                    hidden = merge_partial(
                        self.mixers[i],
                        inter_outputs[i % block_size],
                        inter_lses[i % block_size],
                        partial,
                    )
                    if capture:
                        _, weights = self.mixers[i](sources)
                else:
                    hidden, weights = self.mixers[i](sources)
            if capture:
                trace["inputs"].append(hidden)
                if hidden.requires_grad:
                    hidden.retain_grad()
            branch, token_weights = layer(hidden)
            if c.variant == "baseline":
                hidden = hidden + branch
            else:
                # 저장할 것은 누적 hidden이 아니라 f_i(h_i). block 안에서만 합산.
                partial = branch if partial is None else partial + branch
                if (i + 1) % block_size == 0 or i + 1 == len(self.layers):
                    completed.append(partial)
                    partial = None
            if capture:
                trace["outputs"].append(branch)
                if branch.requires_grad:
                    branch.retain_grad()
                if weights is not None:
                    trace["depth_weights"].append(weights.detach())
                if token_weights is not None:
                    trace["token_weights"].append(token_weights.detach())
        if c.variant != "baseline":
            hidden, weights = self.mixers[-1](completed)
            if capture:
                trace["depth_weights"].append(weights.detach())
        logits = self.classifier(self.final_norm(hidden[:, -1]))
        return (logits, trace) if capture else logits

    def config_dict(self):
        return asdict(self.config)


def matched_model(config, seed):
    """variant마다 backbone 초기값은 같고, mixer만 zero-init으로 추가된다."""
    torch.manual_seed(seed)
    baseline = RecallTransformer(Config(**{**asdict(config), "variant": "baseline"}))
    torch.manual_seed(seed)
    model = RecallTransformer(config)
    missing, unexpected = model.load_state_dict(baseline.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("mixers.") for k in missing)
    return model
