"""Pippo 원본을 열고, 작은 tensor로 그 안의 실제 연산을 따라간다.

새 모델 구현이 아니다. external/pippo의 원본 class를 직접 호출한다.
patch/control/block은 연산을 보는 진단이고, fica는 기존 02 smoke checkpoint를 읽는다.

python pippo_walkthrough.py --part files
CUDA_VISIBLE_DEVICES=0 python pippo_walkthrough.py --part patch
CUDA_VISIBLE_DEVICES=0 python pippo_walkthrough.py --part control
CUDA_VISIBLE_DEVICES=0 python pippo_walkthrough.py --part block
CUDA_VISIBLE_DEVICES=0 python pippo_walkthrough.py --part fica

원본 forward 안으로 들어가려면 --break를 붙이고 pdb에서 s/n/p를 사용한다.
"""

import argparse
import ast
import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "external/pippo"
spec = importlib.util.spec_from_file_location("fica_prepare", ROOT / "00_prepare_nersemble.py")
L00 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L00)


def original_source():
    """학습 본편과 같은 commit의 전체 checkout인지, 공유 파일이 동일한지 확인한다."""
    if not (SOURCE / ".git").is_dir():
        raise FileNotFoundError("GUIDE.md의 'Pippo 원본을 직접 읽기'에 있는 clone/checkout 명령을 사용하세요.")
    revision = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    if revision != L00.SOURCE_REVISIONS["pippo"]:
        raise ValueError(f"본편과 다른 Pippo commit: {revision}")
    subprocess.run(["git", "-C", str(SOURCE), "diff", "--exit-code", "--quiet", "HEAD", "--"], check=True)
    for name in L00.SOURCE_FILES["pippo"]:
        cached = L00.CACHE / "reference/pippo" / name
        if cached.is_file() and L00.sha256(SOURCE / name) != L00.sha256(cached):
            raise ValueError(f"본편 cache와 원본 checkout이 다르다: {name}")
    # namespace package의 최초 import 전에 원본 checkout을 우선한다.
    sys.path.insert(0, str(SOURCE))
    from latent_diffusion.models.dit import DiTBlock
    if not Path(inspect.getfile(DiTBlock)).resolve().is_relative_to(SOURCE):
        raise RuntimeError("다른 latent_diffusion이 이미 import되어 있다. 새 Python process에서 실행하세요.")
    print("원본:", SOURCE, "| commit:", revision)
    return revision


def files():
    """거대한 Pippo class를 import하지 않고도 실제 읽을 정의의 줄을 찾는다."""
    targets = {
        "latent_diffusion/blocks/patch.py": {"PatchifyPS", "UnpatchifyPS"},
        "latent_diffusion/blocks/timestep.py": {"TimestepEmbedder"},
        "latent_diffusion/models/control_mlp.py": {"ControlEncoderLatent", "ZeroControlBlock", "ControlMLP"},
        "latent_diffusion/models/dit.py": {"DiTBlock", "DiT"},
        "latent_diffusion/models/pippo.py": {"get_conds", "forward"},
        "latent_diffusion/losses.py": {"DenoisingLoss"},
    }
    for name, symbols in targets.items():
        for node in ast.walk(ast.parse((SOURCE / name).read_text())):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in symbols:
                print(f"  {name}:{node.lineno}  {node.name}")


def trace(label):
    """Hook은 결과를 출력할 뿐 원본의 입력/출력을 바꾸지 않는다."""
    def hook(module, inputs, output):
        def shape(value):
            return list(value.shape) if isinstance(value, torch.Tensor) else [shape(x) for x in value] if isinstance(value, (tuple, list)) else type(value).__name__
        print(f"  {label}: {shape(inputs)} -> {shape(output)}")
    return hook


def forward(module, *args, pause=False, **kwargs):
    if pause:
        import pdb
        # 원본의 첫 줄에서 멈춘다. n=다음 줄, s=호출 내부, p x.shape=값 확인, c=계속.
        return pdb.Pdb().runcall(module.forward, *args, **kwargs)
    return module(*args, **kwargs)


def patch(device, pause):
    from latent_diffusion.blocks.patch import PatchifyPS
    # FiCA packing 이후의 작은 latent 예: 16ch, 8x8. 실제 본 설정은 16ch, 32x32.
    latent = torch.arange(16*8*8, device=device).float().reshape(1,16,8,8)/1024
    module = PatchifyPS(img_size=8, patch_size=2, in_channels=16, out_channels=64, flatten=True).to(device)
    tokens = forward(module, latent, pause=pause)
    assert tokens.shape == (1,16,64)
    print("patch:", list(latent.shape), "->", list(tokens.shape))
    print("16 tokens는 4x4 위치다. 64차원은 학습된 feature이며 XYZ 좌표가 아니다.")
    return {"input": list(latent.shape), "tokens": list(tokens.shape)}


def control(device, pause):
    from latent_diffusion.models.control_mlp import ControlMLP
    module = ControlMLP(64,64,depth=2,control_encoder_latent=True).to(device)
    dense = torch.randn(1,64,8,8,device=device)
    time = torch.tensor([500.], device=device)
    handle = module.control_encoder.register_forward_hook(trace("ControlEncoderLatent"))
    controls = forward(module, dense, time, pause=pause)
    handle.remove()
    assert len(controls)==2 and all(x.shape==(1,16,64) for x in controls)
    assert all(torch.count_nonzero(x)==0 for x in controls)
    optimizer = torch.optim.SGD(module.parameters(), lr=.01)
    # 0 출력에 대한 0 loss는 gradient가 없다. 진단용 nonzero target을 명시한다.
    loss = (controls[0]-1).square().mean()
    loss.backward()
    encoder_gradient = float(module.control_encoder.conv_block[0].weight.grad.abs().sum())
    zero_linear_gradient = float(module.zero_blocks[0].zero_linear.weight.grad.abs().sum())
    assert encoder_gradient==0 and zero_linear_gradient>0
    optimizer.step()
    changed = float(module(dense,time)[0].detach().abs().max())
    assert changed>0
    print("control: depth별 control list", [list(x.shape) for x in controls])
    print("첫 backward: encoder gradient =", encoder_gradient, "| zero_linear gradient =", zero_linear_gradient)
    print("첫 update 후 control max =", changed)
    return {"initially_zero": True, "encoder_gradient_first": encoder_gradient,
            "zero_linear_gradient_first": zero_linear_gradient, "control_after_update_max": changed}


def block(device, pause):
    from latent_diffusion.models.dit import DiT
    # 원본 DiT.initialize_weights가 수행하는 zero-gate 초기화까지 그대로 사용한다.
    owner = DiT(img_size=8,patch_size=2,in_channels=16,dim=64,depth=2,num_heads=4,
                cond_attn_enabled=False,pos_embed_kind="old").to(device)
    module = owner.blocks[0]
    x = torch.randn(1,16,64,device=device)
    condition = torch.randn(1,64,device=device)
    dense = torch.randn_like(x)
    handles = [module.in_proj.register_forward_hook(trace("MLP+QKV joint projection")),
               module.q_norm.register_forward_hook(trace("Q head normalization")),
               module.cond_adaln.register_forward_hook(trace("global shift/scale/gates"))]
    initial = forward(module,x,condition,control=dense,pause=pause)
    for handle in handles:
        handle.remove()
    assert torch.equal(initial,x)
    optimizer = torch.optim.SGD(module.parameters(),lr=.01)
    initial.square().mean().backward()
    optimizer.step()
    with torch.no_grad():
        with_control = module(x,condition,control=dense)
        without_control = module(x,condition,control=torch.zeros_like(dense))
        effect = float((with_control-without_control).abs().max())
    assert effect>0
    print("block: zero gate인 최초 출력은 x와 같다. gate update 후 control 영향 =", effect)
    print("원본 마지막 식은 x + x_attn + x_mlp다. Control은 MLP branch에 더해진다.")
    return {"initial_identity": True, "control_effect_after_gate_update_max": effect}


def fica(device, pause):
    """기존 실제 02 smoke 결과를 원본 checkout의 동일 class로 읽어 연결을 확인한다."""
    module = L00.lesson(2)
    config = L00.read_config(smoke=True)
    data = L00.PreparedData(L00.work_root(config)/"data")
    cache = module.read_cache(data,data.samples("train")[0],device)
    model = module.UVFlowModel(config).to(device).eval()
    checkpoint = L00.restore(L00.work_root(config)/"checkpoints/02/latest.pt",model,config,data)
    handles = [model.x_embed.register_forward_hook(trace("target patch embedding")),
               model.control.control_encoder.register_forward_hook(trace("partial latent encoder")),
               model.control.register_forward_hook(trace("depth별 spatial controls")),
               model.blocks[0].in_proj.register_forward_hook(trace("첫 DiT MLP+QKV")),
               model.output.register_forward_hook(trace("velocity patch output"))]
    latent = torch.randn_like(cache["texture"])[None]
    time = torch.tensor([.5],device=device)
    args = (latent,time,cache["clip"][None],cache["control"][None],torch.tensor([0],device=device))
    with torch.no_grad():
        output = forward(model,*args,pause=pause)
        for handle in handles:
            handle.remove()
        other = model(latent,time,args[2],torch.zeros_like(args[3]),args[4])
    effect = float((output-other).abs().max())
    assert output.shape==latent.shape and torch.isfinite(output).all() and effect>0
    print("fica: checkpoint step",checkpoint["step"],"| velocity",list(output.shape),"| control 영향",effect)
    return {"real_cached_partial": True, "checkpoint_step": checkpoint["step"],
            "output": list(output.shape), "control_effect_max": effect}


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part",choices=("files","patch","control","block","fica","all"),default="files")
    parser.add_argument("--break",dest="pause",action="store_true",help="원본 forward 첫 줄에서 pdb로 멈춘다")
    args = parser.parse_args()
    revision = original_source()
    if args.part=="files":
        files()
        return
    device = L00.setup_device("cuda:0")
    report = {"source":str(SOURCE),"revision":revision,"device":str(device),"parts":{}}
    for name, part in (("patch",patch),("control",control),("block",block),("fica",fica)):
        if args.part in ("all",name):
            report["parts"][name]=part(device,args.pause)
    L00.save_json(L00.CACHE/"pippo_walkthrough"/f"{args.part}.json",report)


if __name__=="__main__":
    main()
