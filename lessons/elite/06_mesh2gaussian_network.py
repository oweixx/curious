"""06 — Driving encoder + conditioned Song U-Net + geometry/appearance U-Net.

원본 비교:
 https://github.com/kaist-ami/ELITE/blob/58a7a71dc3e922f589f87882de46154594da0bc3/src/models/mesh_unet.py
 https://github.com/kaist-ami/ELITE/blob/58a7a71dc3e922f589f87882de46154594da0bc3/src/nn/unet_gs.py

계산을 읽기 쉽게 PyTorch 기본 layer로 다시 작성했다. 6개 resolution, 각 block의 driving
conditioning, bottleneck/16x16 self-attention, 모든 skip, 6-level spatial-bias heads를 포함한다.
이번 network의 state_dict는 공개 MGPM checkpoint와 호환되지 않는다. 초기화/weight-norm
parameter 저장 형식이 다르며, 한 사람 scratch 학습용이다. Pretrained prior를 사용했다고
부르지 않는다. Native checkpoint의 resume/inference는 08/09에서 strict loading한다.

CUDA_VISIBLE_DEVICES=2 python 06_mesh2gaussian_network.py
출력: 실제 입력에 대한 layer shape 기록과 trainable parameter 수. 아직 학습 전이다.
"""

import argparse
import importlib.util
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

spec = importlib.util.spec_from_file_location("elite_03_entry", Path(__file__).with_name("03_inspect_tracking.py"))
L03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L03)


def wn_linear(cin,cout):
    layer = nn.Linear(cin,cout)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return weight_norm(layer,dim=0)


class DrivingEncoder(nn.Module):
    """100 expression + quaternion pose. 각 성분을 별도로 projection해서 128차원으로 합친다."""
    def __init__(self,embedding=128):
        super().__init__()
        if embedding%4:
            raise ValueError("embedding은 4의 배수여야 한다.")
        def project(cin,cout):
            return nn.Sequential(wn_linear(cin,cout),nn.LeakyReLU(.2))
        self.expr = project(100,embedding)
        self.rottrans = project(7,embedding//4)
        self.neck = project(4,embedding//4)
        self.jaw = project(4,embedding//4)
        self.eyes = project(8,embedding//4)
        self.mlp = nn.Sequential(wn_linear(embedding*2,embedding*2),nn.LeakyReLU(.2),
                                 wn_linear(embedding*2,embedding*2),nn.LeakyReLU(.2),
                                 wn_linear(embedding*2,embedding))

    def forward(self,p):
        q = L03.lesson(5).axis_angle_to_quaternion
        pieces = [self.expr(p["expr"]),self.jaw(q(p["jaw_pose"])),
                  self.eyes(torch.cat((q(p["eyes_pose"][:,:3]),q(p["eyes_pose"][:,3:])),1)),
                  self.rottrans(torch.cat((q(p["rotation"]),p["translation"]),1)),self.neck(q(p["neck_pose"]))]
        return self.mlp(torch.cat(pieces,1))


def norm(channels):
    groups = min(32,max(1,channels//4))
    while channels%groups:
        groups-=1
    return nn.GroupNorm(groups,channels,eps=1e-6)


class ConditionedBlock(nn.Module):
    """Residual convolution + additive driving conditioning + optional self-attention.

    Original SongUNet config는 adaptive_scale=False다. Embedding은 spatial broadcast로
    conv0에 더한다. 원본의 연속된 두 번째 norm/SiLU도 생략하지 않는다.
    """
    def __init__(self,cin,cout,embedding,resize=None,attention=False,dropout=.1):
        super().__init__()
        self.in_channels,self.out_channels = cin,cout
        self.resize,self.attention,self.dropout = resize,attention,dropout
        self.norm0,self.norm1 = norm(cin),norm(cout)
        self.conv0,self.conv1 = nn.Conv2d(cin,cout,3,padding=1),nn.Conv2d(cout,cout,3,padding=1)
        self.affine = nn.Linear(embedding,cout)
        self.skip = nn.Conv2d(cin,cout,1) if cin!=cout or resize else nn.Identity()
        nn.init.xavier_uniform_(self.conv0.weight)
        nn.init.zeros_(self.conv0.bias)
        nn.init.xavier_uniform_(self.conv1.weight,gain=1e-5)
        nn.init.zeros_(self.conv1.bias)
        nn.init.xavier_uniform_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)
        if attention:
            self.norm2 = norm(cout)
            self.qkv = nn.Conv2d(cout,3*cout,1)
            self.proj = nn.Conv2d(cout,cout,1)
            nn.init.xavier_uniform_(self.qkv.weight,gain=math.sqrt(.2))
            nn.init.zeros_(self.qkv.bias)
            nn.init.xavier_uniform_(self.proj.weight,gain=1e-5)
            nn.init.zeros_(self.proj.bias)

    def resample(self,x):
        # [1,1] separable box filter와 대응: down은 평균, up은 nearest replication.
        if self.resize=="down":
            return F.avg_pool2d(x,2)
        if self.resize=="up":
            return F.interpolate(x,scale_factor=2,mode="nearest")
        return x

    def forward(self,x,embedding):
        original = self.skip(self.resample(x))
        x = self.conv0(self.resample(F.silu(self.norm0(x))))
        x = F.silu(self.norm1(x+self.affine(embedding)[:,:,None,None]))
        x = F.silu(self.norm1(x))
        x = (original+self.conv1(F.dropout(x,p=self.dropout,training=self.training)))*math.sqrt(.5)
        if self.attention:
            b,c,h,w = x.shape
            # 한 attention head. QK^T는 pixel 간 관계이며 driving 정보는 이미 block에 들어왔다.
            q,k,v = self.qkv(self.norm2(x)).reshape(b,3,c,h*w).unbind(1)
            weights = torch.softmax(torch.bmm(q.transpose(1,2),k)/math.sqrt(c),dim=-1)
            attended = torch.bmm(v,weights.transpose(1,2)).reshape(b,c,h,w)
            x = (x+self.proj(attended))*math.sqrt(.5)
        return x


class ConditionedSongUNet(nn.Module):
    def __init__(self,size,model_channels=32,multipliers=(1,2,2,2,2,2),blocks=1,embedding=128,attention_sizes=(16,)):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(embedding,model_channels*2),nn.SiLU(),
                                       nn.Linear(model_channels*2,model_channels*2),nn.SiLU())
        self.stem = nn.Conv2d(6,model_channels,3,padding=1)
        self.encoder,self.decoder = nn.ModuleList(),nn.ModuleList()
        channels = model_channels
        skip_channels = [channels]
        for level,mult in enumerate(multipliers):
            resolution = size//(2**level)
            if level:
                block = ConditionedBlock(channels,channels,model_channels*2,resize="down")
                self.encoder.append(block)
                skip_channels.append(channels)
            for _ in range(blocks):
                out = model_channels*mult
                self.encoder.append(ConditionedBlock(channels,out,model_channels*2,attention=resolution in attention_sizes))
                channels=out
                skip_channels.append(channels)
        self.middle = nn.ModuleList([ConditionedBlock(channels,channels,model_channels*2,attention=True),
                                     ConditionedBlock(channels,channels,model_channels*2)])
        self.decoder_uses_skip=[]
        for level,mult in reversed(list(enumerate(multipliers))):
            resolution = size//(2**level)
            if level!=len(multipliers)-1:
                self.decoder.append(ConditionedBlock(channels,channels,model_channels*2,resize="up"))
                self.decoder_uses_skip.append(False)
            for i in range(blocks+1):
                incoming = channels+skip_channels.pop()
                channels = model_channels*mult
                self.decoder.append(ConditionedBlock(incoming,channels,model_channels*2,
                    attention=i==blocks and resolution in attention_sizes))
                self.decoder_uses_skip.append(True)
        if skip_channels:
            raise ValueError("skip graph가 맞지 않는다.")
        self.out = nn.Sequential(norm(channels),nn.SiLU(),nn.Conv2d(channels,16,3,padding=1))
        nn.init.normal_(self.out[-1].weight,std=1e-3)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self,x,driving):
        embedding = self.embedding(driving)
        x=self.stem(x)
        skips=[x]
        for block in self.encoder:
            x=block(x,embedding)
            skips.append(x)
        for block in self.middle:
            x=block(x,embedding)
        for block,use_skip in zip(self.decoder,self.decoder_uses_skip):
            if use_skip:
                x=torch.cat((x,skips.pop()),1)
            x=block(x,embedding)
        return self.out(x)


class SpatialConv(nn.Module):
    """Weight-normalized conv + pixel별 learned bias. 각 UV 위치의 특성을 학습한다."""
    def __init__(self,cin,cout,size,transpose=False,kernel=4,stride=2,padding=1):
        super().__init__()
        layer = nn.ConvTranspose2d(cin,cout,kernel,stride,padding,bias=False) if transpose else nn.Conv2d(cin,cout,kernel,stride,padding,bias=False)
        nn.init.xavier_uniform_(layer.weight,gain=nn.init.calculate_gain("leaky_relu",.2))
        self.conv = weight_norm(layer,dim=1 if transpose else 0)
        self.bias = nn.Parameter(torch.zeros(cout,size,size))

    def forward(self,x):
        return self.conv(x)+self.bias[None]


class EnhancementHead(nn.Module):
    """여기서 enhancement는 UV refinement head다. 제외한 diffusion enhancer와 다르다."""
    def __init__(self,size,out_channels,base=16):
        super().__init__()
        channels=[base*2**i for i in range(6)]
        self.down,self.up=nn.ModuleList(),nn.ModuleList()
        incoming=16
        for level,outgoing in enumerate(channels):
            self.down.append(nn.Sequential(SpatialConv(incoming,outgoing,size//2**(level+1)),nn.LeakyReLU(.2)))
            incoming=outgoing
        for level in reversed(range(6)):
            outgoing=channels[level-1] if level else base
            self.up.append(nn.Sequential(SpatialConv(incoming,outgoing,size//2**level,transpose=True),nn.LeakyReLU(.2)))
            incoming=outgoing*2 if level else outgoing+16
        self.out=SpatialConv(incoming,out_channels,size,kernel=1,stride=1,padding=0)
        # 작은 nonzero gain으로 초기 geometry를 안정화하면서 upstream gradient를 유지한다.
        with torch.no_grad():
            self.out.conv.parametrizations.weight.original0.mul_(.001)

    def forward(self,x):
        original=x
        skips=[]
        for down in self.down:
            x=down(x)
            skips.append(x)
        for index,up in enumerate(self.up):
            x=up(x)
            x=torch.cat((x,skips[-2-index] if index<5 else original),1)
        return self.out(x)


class Mesh2Gaussian(nn.Module):
    def __init__(self,config):
        super().__init__()
        network=config["network"]
        size=config["uv_size"]
        if size<64 or size%64:
            raise ValueError("6-level head에는 UV size>=64, 64의 배수가 필요하다.")
        self.driving=DrivingEncoder(network["embedding"])
        self.unet=ConditionedSongUNet(size,network["model_channels"],tuple(network["channel_mult"]),
                                    network["num_blocks"],network["embedding"],tuple(network["attention_sizes"]))
        self.geometry_head=EnhancementHead(size,13)
        self.appearance_head=EnhancementHead(size,3)
        with torch.no_grad():
            self.geometry_head.out.bias[3:5].fill_(L03.lesson(5).initial_scale_logit(size))

    def forward(self,uv_input,parameters):
        driving=self.driving(parameters)
        features=self.unet(uv_input,driving)
        return self.geometry_head(features),self.appearance_head(features)


def parameter_batch(scene,ids,device):
    index=torch.as_tensor(ids,dtype=torch.long)
    return {k:scene["parameters"][k][index].to(device) for k in
            ("expr","rotation","translation","neck_pose","jaw_pose","eyes_pose")}


def load_config(path=None):
    import json
    return json.loads((Path(path) if path else L03.ROOT/"avatar_config.json").read_text(encoding="utf-8"))


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    L03.add_arguments(parser)
    parser.add_argument("--config",type=Path)
    args=parser.parse_args()
    scene=L03.read_scene(args.sequence,args.run)
    uv=L03.lesson(5).read_uv(scene)
    config=load_config(args.config)
    if config["uv_size"]!=uv["uv_size"]:
        raise ValueError("config와 04 UV resolution을 맞추세요.")
    torch.manual_seed(config["seed"])
    model=Mesh2Gaussian(config).to(args.device).eval()
    shape_log=[]
    feature_log={}
    handles=[]
    def record(name):
        def hook(module,inputs,output):
            if isinstance(output,torch.Tensor):
                shape_log.append({"layer":name,"shape":list(output.shape)})
                if name=="unet.middle.0":
                    feature_log[name]=output[0,:4].detach().cpu()
        return hook
    for name,module in model.named_modules():
        if isinstance(module,(ConditionedBlock,SpatialConv,DrivingEncoder)):
            handles.append(module.register_forward_hook(record(name)))
    with torch.no_grad():
        input_uv=torch.cat((uv["texture"]*2-1,uv["normalized_xyz"]),0)[None].to(args.device)
        geo,app=model(input_uv,parameter_batch(scene,[scene["split"]["train"][0]],args.device))
    for handle in handles:
        handle.remove()
    out=L03.output_dir(6,args.sequence,args.run)
    L03.save_tensor(out/"initial_network_outputs.pt",{"geometry":geo.cpu(),"appearance":app.cpu(),
                                                     "bottleneck_features":feature_log})
    for name,feature in feature_log.items():
        tiles=[]
        for channel in feature:
            normalized=(channel-channel.min())/(channel.max()-channel.min()).clamp_min(1e-8)
            tiles.append(normalized[None].expand(3,-1,-1))
        # 4 feature channel은 각자 범위를 정규화한다. 같은 color가 같은 physical value는 아니다.
        image=torch.cat(tiles,-1)[None]
        image=F.interpolate(image,scale_factor=8,mode="nearest")[0]
        L03.save_image(out/"initial_bottleneck_features.png",image)
    L03.save_json(out/"network_shapes.json",{"trainable_parameters":sum(p.numel() for p in model.parameters()),
        "geometry_output":list(geo.shape),"appearance_output":list(app.shape),"layers":shape_log,
        "checkpoint_format":"single-identity scratch reimplementation; public MGPM incompatible"})
    for row in shape_log:
        print(row["layer"],row["shape"])
    print("06 완료:",out,"| 학습은 아직 수행하지 않았다.")


if __name__=="__main__":
    main()
