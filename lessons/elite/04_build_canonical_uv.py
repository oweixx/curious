"""04 — 실제 FLAME 표면을 UV pixel로 옮기고 train RGB로 texture를 만든다.

1) UV triangle rasterization은 어느 face인지 찾는 용도로만 이용한다.
2) barycentric weight는 식을 직접 계산한다. P_uv = sum_i w_i * vertex_i.
3) 각 frame에서 개인 tracking mesh의 UV 위치를 camera에 투영한다.
4) z-buffer, front-facing, alpha를 통과한 관측만 가중 평균한다.
5) Gaussian 배치용 mesh의 TBN은 frame마다 새로 계산한다.

CUDA_VISIBLE_DEVICES=2 python 04_build_canonical_uv.py --uv-size 512
결과: uv.pt, XYZ/texture/normal/coverage/face-index 시각화.
Texture는 관측된 색이며 intrinsic albedo가 아니다. Validation RGB는 사용하지 않는다.
"""

import argparse
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("elite_03_entry", Path(__file__).with_name("03_inspect_tracking.py"))
L03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L03)


def uv_lookup(vt, ft, size, context):
    import torch
    import nvdiffrast.torch as dr
    # nvdiffrast rows는 아래에서 위로 진행한다. flip 후 row=0은 v=1이 된다.
    clip = torch.cat((vt*2-1, torch.zeros_like(vt[:, :1]), torch.ones_like(vt[:, :1])), -1)
    rast, _ = dr.rasterize(context, clip[None].contiguous(), ft.int().contiguous(), [size, size])
    face = rast[0, ..., 3].long().flip(0) - 1  # 0은 background -> -1
    covered = face >= 0
    if not covered.any():
        raise ValueError("UV rasterization 결과가 비어 있다. UV 범위/triangle winding을 확인하세요.")
    row, col = torch.meshgrid(torch.arange(size, device=vt.device), torch.arange(size, device=vt.device), indexing="ij")
    # 얇은 UV triangle에서도 가중치를 안정적으로 계산하도록 이 계산만 float64로 한다.
    query = torch.stack(((col.double()+.5)/size, 1-(row.double()+.5)/size), dim=-1)
    tri = vt.double()[ft[face.clamp_min(0)]]  # [H,W,3,2]. Background는 아래에서 마스킹한다.
    a, b, c = tri.unbind(-2)
    e0, e1, q = b-a, c-a, query-a
    determinant = e0[..., 0]*e1[..., 1] - e0[..., 1]*e1[..., 0]
    safe = torch.where(determinant.abs() > 1e-12, determinant, torch.ones_like(determinant))
    wb = (q[..., 0]*e1[..., 1] - q[..., 1]*e1[..., 0])/safe
    wc = (e0[..., 0]*q[..., 1] - e0[..., 1]*q[..., 0])/safe
    bary = torch.stack((1-wb-wc, wb, wc), dim=-1)
    usable = covered & (determinant.abs()>1e-12)
    if not usable.any() or not torch.isfinite(bary[usable]).all():
        raise ValueError("퇴화하지 않은 UV triangle의 유한한 barycentric 좌표가 없다.")

    # CUDA rasterizer는 coverage를 판정할 때 vertex를 1/16 pixel 격자에 snap한다.
    # 따라서 선택된 face의 '원래' 경계 밖에 pixel center가 살짝 있을 수 있다.
    # barycentric 값의 오차는 triangle 두께에 따라 커지므로 pixel 거리로 검사한다.
    opposite_edges = torch.stack((c-b, a-c, b-a), dim=-2)
    edge_lengths = opposite_edges.norm(dim=-1).clamp_min(1e-20)
    signed_edge_distance_px = bary * determinant.abs()[..., None] / edge_lengths * size
    tolerance_px = 1/16 + 4*torch.finfo(vt.dtype).eps*size
    max_outside_px = float((-signed_edge_distance_px[usable].min()).clamp_min(0))
    if max_outside_px > tolerance_px:
        raise ValueError(
            f"UV face/pixel 대응이 {max_outside_px:.6f} pixel 어긋났다 "
            f"(coverage snap 허용 거리 {tolerance_px:.6f}). "
            "UV row flip/face index/좌표계를 확인하세요."
        )
    # 원래 triangle 밖의 center는 valid에서 제외한다. 음수 가중치로 표면 밖을 보간하지 않는다.
    boundary_excluded = usable & (bary.min(dim=-1).values < -1e-10)
    valid = usable & ~boundary_excluded
    if not valid.any():
        raise ValueError("원래 UV triangle 내부에 있는 pixel center가 없다.")
    # 여기서는 float64 산술의 극소 오차만 제거한다. 경계 밖 texel을 강제로 clamp하지 않는다.
    bary = bary.clamp_min(0)
    bary = bary / bary.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    reconstructed = (bary[..., None]*tri).sum(-2)
    if (reconstructed[valid]-query[valid]).abs().max() > 1e-5:
        raise ValueError("UV barycentric reconstruction 오류")
    bary = torch.where(valid[..., None], bary, torch.zeros_like(bary))
    face = torch.where(valid, face, torch.full_like(face, -1))
    print(f"04 UV lookup: covered={int(covered.sum())}, valid={int(valid.sum())}, "
          f"boundary_excluded={int(boundary_excluded.sum())}, max_outside={max_outside_px:.6f}px")
    return {"face": face, "bary": bary.to(vt.dtype), "valid": valid,
            "query_uv": query.to(vt.dtype), "boundary_excluded": boundary_excluded}


def interpolate_vertices(vertices, faces, lookup):
    """[B,V,C] -> [B,H,W,C]. 이 gather/sum은 vertices에 대해 미분 가능하다."""
    corners = faces[lookup["face"].clamp_min(0)]
    values = vertices[:, corners]  # [B,H,W,3,C]
    return (values * lookup["bary"][None, ..., None]).sum(-2) * lookup["valid"][None, ..., None]


def face_tbn(vertices, vt, faces, ft):
    """dP/du를 구하고 right-handed [T,B,N]을 만든다. 결과 [B,F,3,3].

    UV mirrored island에도 det(R)=+1인 회전이 필요하다. B = cross(N,T)로 고정한다.
    ELITE의 [tangent,-bitangent,normal]과 같은 handedness다. Degenerate face는
    renderer에 NaN을 넘기지 않게 fallback frame을 쓰며 04에서 개수를 보고한다.
    """
    import torch
    import torch.nn.functional as F
    p0, p1, p2 = vertices[:, faces].unbind(-2)
    uv0, uv1, uv2 = vt[ft].unbind(-2)
    e1, e2 = p1-p0, p2-p0
    d1, d2 = uv1-uv0, uv2-uv0
    determinant = d1[:, 0]*d2[:, 1] - d1[:, 1]*d2[:, 0]
    safe = torch.where(determinant.abs()>1e-12, determinant, torch.ones_like(determinant))
    raw_t = (e1*d2[None, :, 1:2] - e2*d1[None, :, 1:2])/safe[None, :, None]
    raw_n = torch.cross(e1, e2, dim=-1)
    bad = (raw_n.norm(dim=-1)<1e-10) | (raw_t.norm(dim=-1)<1e-10)
    n = F.normalize(raw_n, dim=-1)
    fallback_n = torch.zeros_like(n)
    fallback_n[..., 2] = 1
    n = torch.where(bad[..., None], fallback_n, n)
    t = raw_t - (raw_t*n).sum(-1, keepdim=True)*n
    # Gram-Schmidt에 필요한 fallback: normal과 가장 덜 평행한 axis를 선택한다.
    axis = F.one_hot(n.abs().argmin(-1), 3).to(n.dtype)
    fallback_t = F.normalize(torch.cross(axis, n, dim=-1), dim=-1)
    t = torch.where((t.norm(dim=-1)<1e-10)[..., None], fallback_t, F.normalize(t, dim=-1))
    b = F.normalize(torch.cross(n, t, dim=-1), dim=-1)
    return torch.stack((t, b, n), dim=-1), bad


def image_projection_matrix(K, h, w, near=.01, far=10.):
    """OpenGL camera -> clip. Rasterizer flip(0) 후 image y 방향이 맞는다."""
    import torch
    p = torch.zeros(4, 4, device=K.device)
    p[0, 0], p[1, 1] = 2*K[0, 0]/w, 2*K[1, 1]/h
    p[0, 2], p[1, 2] = 1-2*K[0, 2]/w, 2*K[1, 2]/h-1
    p[2, 2], p[2, 3], p[3, 2] = -(far+near)/(far-near), -2*far*near/(far-near), -1
    return p


def image_depth(vertices, faces, K, w2c, h, w, context):
    import torch
    import nvdiffrast.torch as dr
    cam = vertices @ w2c[:3, :3].T + w2c[:3, 3]
    p = image_projection_matrix(K, h, w)
    homogeneous = torch.cat((cam, torch.ones_like(cam[..., :1])), -1)
    clip = homogeneous @ p.T
    rast, _ = dr.rasterize(context, clip.contiguous(), faces.int().contiguous(), [h, w])
    depth, _ = dr.interpolate((-cam[..., 2:3]).contiguous(), rast, faces.int().contiguous())
    return depth[0].permute(2, 0, 1).flip(1), (rast[0, ..., 3]>0).flip(0)[None].float()


def sample_image(image, pixels, h, w, mode="bilinear"):
    import torch.nn.functional as F
    # align_corners=False: grid=-1은 pixel center가 아니라 이미지 왼쪽 경계다.
    grid = pixels.clone()
    grid[..., 0] = 2*grid[..., 0]/w-1
    grid[..., 1] = 2*grid[..., 1]/h-1
    return F.grid_sample(image[None], grid[None], mode=mode, padding_mode="zeros", align_corners=False)[0].permute(1, 2, 0)


def fill_unseen(texture, observed, valid):
    """같은 UV island 안에서만 nearest 관측으로 채운다. 관측 confidence는 늘리지 않는다."""
    import numpy as np
    import torch
    from scipy.ndimage import label, distance_transform_edt
    output = texture.cpu().numpy().copy()
    seen, mask = observed.cpu().numpy(), valid.cpu().numpy()
    islands, count = label(mask)
    for i in range(1, count+1):
        island = islands == i
        known = seen & island
        if not known.any():
            continue
        _, nearest = distance_transform_edt(~known, return_indices=True)
        missing = island & ~known
        output[missing] = output[nearest[0][missing], nearest[1][missing]]
    # Conv가 UV island border에서도 문맥을 읽도록 invalid 공간만 nearest valid로 padding.
    _, nearest = distance_transform_edt(~mask, return_indices=True)
    output[~mask] = output[nearest[0][~mask], nearest[1][~mask]]
    return torch.from_numpy(np.asarray(output)).to(texture.device)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    L03.add_arguments(parser)
    parser.add_argument("--uv-size", type=int, default=512)
    parser.add_argument("--texture-frames", type=int, default=128, help="train에서 균등하게 선택; 0이면 train 전체")
    args = parser.parse_args()
    if args.uv_size < 64 or args.uv_size & (args.uv_size-1) or args.texture_frames < 0:
        parser.error("uv-size는 64 이상 2의 거듭제곱, texture-frames는 0 이상")
    import torch
    import numpy as np
    import nvdiffrast.torch as dr
    from PIL import Image
    scene = L03.read_scene(args.sequence, args.run)
    out = L03.output_dir(4, args.sequence, args.run)
    if (out/"uv.pt").exists():
        raise FileExistsError("04 결과가 이미 있다. 해상도를 바꿀 때 기존 이후 학습도 별도 보관하세요.")
    geometry = L03.FrameGeometry(scene, args.device)
    context = dr.RasterizeCudaContext(device=args.device)
    lookup = uv_lookup(geometry.vt, geometry.ft, args.uv_size, context)
    enc = scene["encoding_vertices"].to(args.device)
    xyz = interpolate_vertices(enc, geometry.faces, lookup)[0]
    tbn, bad = face_tbn(enc, geometry.vt, geometry.faces, geometry.ft)
    normal = tbn[0, lookup["face"].clamp_min(0), :, 2]*lookup["valid"][..., None]
    tangent = tbn[0, lookup["face"].clamp_min(0), :, 0]*lookup["valid"][..., None]
    bitangent = tbn[0, lookup["face"].clamp_min(0), :, 1]*lookup["valid"][..., None]
    good=tbn[0, ~bad[0]]
    if len(good) and ((torch.linalg.det(good)-1).abs().max()>1e-4):
        raise ValueError("TBN이 right-handed 회전이 아니다. UV mirror/winding을 확인하세요.")
    
    h, w = scene["manifest"]["height"], scene["manifest"]["width"]
    K, w2c = scene["K"].to(args.device), scene["w2c"].to(args.device)
    color_sum, weight_sum = torch.zeros_like(xyz), torch.zeros_like(xyz[..., :1])
    train = scene["split"]["train"]
    if args.texture_frames and len(train)>args.texture_frames:
        train = [train[j] for j in np.unique(np.linspace(0, len(train)-1, args.texture_frames).astype(int))]
    sequence = Path(scene["paths"]["sequence"])
    camera_center = torch.linalg.inv(w2c)[:3, 3]
    with torch.no_grad():
        for slot, i in enumerate(train):
            record = scene["manifest"]["frames"][i]
            rgb = torch.from_numpy(np.asarray(Image.open(sequence/record["image"]).convert("RGB")).copy()).float().to(args.device)/255
            alpha = torch.from_numpy(np.asarray(Image.open(sequence/record["alpha_png"]).convert("L")).copy()).float().to(args.device)/255
            keep = torch.from_numpy(np.asarray(Image.open(L03.output_dir(3,args.sequence,args.run)/"head_masks"/f"{i:06d}.png")).copy()).float().to(args.device)/255
            vertices = geometry.forward([i], personal=True)
            points = interpolate_vertices(vertices, geometry.faces, lookup)[0]
            pixel, depth = L03.project(points, K, w2c)
            zbuffer, raster_valid = image_depth(vertices, geometry.faces, K, w2c, h, w, context)
            reference_z = sample_image(zbuffer, pixel, h, w)[..., 0]
            visible = sample_image(raster_valid, pixel, h, w, "nearest")[..., 0]>.5
            visible &= (reference_z-depth).abs() < .003
            visible &= (depth>.01) & lookup["valid"]
            tbn_i, _ = face_tbn(vertices, geometry.vt, geometry.faces, geometry.ft)
            n = tbn_i[0, lookup["face"].clamp_min(0), :, 2]
            view = torch.nn.functional.normalize(camera_center-points, dim=-1)
            facing = (n*view).sum(-1).clamp(0, 1)
            a = sample_image((alpha*keep)[None], pixel, h, w)[..., 0]
            weight = (visible & (facing>.15) & (a>.5)).float()*facing.square()*a
            color_sum += sample_image(rgb.permute(2,0,1), pixel, h, w)*weight[..., None]
            weight_sum += weight[..., None]
            if slot % 20 == 0:
                print(f"04 texture observation {slot}/{len(train)}: train frame {i}")
    observed = weight_sum[..., 0]>1e-6
    if not observed.any():
        raise ValueError("관측 texture가 없다. camera projection/normal orientation/tracking을 확인하세요.")
    texture = torch.where(observed[..., None], color_sum/weight_sum.clamp_min(1e-6), torch.full_like(xyz,.5))
    texture = fill_unseen(texture, observed, lookup["valid"])
    # Scratch 학습용 XYZ channel 통계. 공개 MGPM의 population statistics와 다르다.
    mean = xyz[lookup["valid"]].mean(0)
    std = xyz[lookup["valid"]].std(0).clamp_min(1e-5)
    normalized_xyz = ((xyz-mean)/std)*lookup["valid"][..., None]
    uv = {k:v.cpu() for k,v in lookup.items()}
    uv.update(texture=texture.permute(2,0,1).cpu(), xyz=xyz.permute(2,0,1).cpu(),
              normalized_xyz=normalized_xyz.permute(2,0,1).cpu(), normal=normal.permute(2,0,1).cpu(),
              tangent=tangent.permute(2,0,1).cpu(),bitangent=bitangent.permute(2,0,1).cpu(),
              xyz_mean=mean.cpu(), xyz_std=std.cpu(), observation_weight=weight_sum[...,0].cpu(),
              observed=observed.cpu(), texture_frame_ids=train, uv_size=args.uv_size,
              scene_sha256=L03.lesson(2).sha256(L03.scene_path(args.sequence,args.run)),
              convention="row=0 -> v=1; col=0 -> u=0; pixel centers; xyzw local quaternion")
    L03.save_tensor(out/"uv.pt", uv)
    L03.save_image(out/"texture.png", uv["texture"])
    L03.save_image(out/"normal.png", uv["normal"]*.5+.5)
    L03.save_image(out/"tangent.png", uv["tangent"]*.5+.5)
    L03.save_image(out/"bitangent.png", uv["bitangent"]*.5+.5)
    L03.save_image(out/"coverage.png", observed[None].float())
    L03.save_image(out/"valid.png", lookup["valid"][None].float())
    L03.save_image(out/"xyz_visualization.png", ((xyz-mean)/(std*6)+.5).permute(2,0,1)*lookup["valid"][None])
    # Face ID 색은 시각화용이다. 저장한 실제 face index를 대신하지 않는다.
    ids = lookup["face"].clamp_min(0).float()
    palette = torch.stack(((ids*.618)%1, (ids*.381)%1, (ids*.173)%1))*lookup["valid"][None]
    L03.save_image(out/"face_index.png", palette)
    L03.save_json(out/"summary.json", {"uv_size":args.uv_size, "valid_texels":int(lookup["valid"].sum()),
                   "boundary_excluded_texels":int(lookup["boundary_excluded"].sum()),
                   "observed_fraction":float(observed.sum()/lookup["valid"].sum()),
                   "degenerate_faces":int(bad.sum()), "texture_frame_ids":train,
                   "normalization":"single-identity valid XYZ per-channel mean/std; scratch only"})
    print("04 완료:", out, "| 관측 비율", float(observed.sum()/lookup["valid"].sum()))


if __name__ == "__main__":
    main()
