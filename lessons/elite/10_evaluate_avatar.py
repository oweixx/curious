"""10 — 모든 train/held-out frame을 평가하고 실패 frame과 시간적 오차를 찾는다.

CUDA_VISIBLE_DEVICES=2 python 10_evaluate_avatar.py --experiment avatar
PSNR/foreground PSNR/SSIM/LPIPS/alpha IoU와 temporal residual L1을 저장한다.
Temporal residual = (render_t-render_previous) - (target_t-target_previous).
이 값은 실제 표정 움직임 자체를 flicker로 오해하지 않도록 reference 변화를 빼지만,
optical flow로 정렬한 perceptual metric은 아니다. 새로운 시점/identity 성능을 입증하지 않는다.
"""

import argparse
import csv
import importlib.util
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

spec=importlib.util.spec_from_file_location("elite_03_entry",Path(__file__).with_name("03_inspect_tracking.py"))
L03=importlib.util.module_from_spec(spec)
spec.loader.exec_module(L03)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    L03.add_arguments(parser)
    parser.add_argument("--experiment",default="avatar")
    parser.add_argument("--checkpoint",choices=("best.pt","latest.pt"),default="best.pt")
    parser.add_argument("--split",choices=("all","train","validation"),default="all")
    args=parser.parse_args()
    device=torch.device(args.device)
    torch.cuda.set_device(device)
    scene,checkpoint,model,surface,dataset=L03.lesson(9).load_avatar(args.sequence,args.run,args.experiment,args.checkpoint,device)
    config=checkpoint["config"]
    uv=L03.lesson(5).read_uv(scene)
    loss=L03.lesson(7).AvatarLoss(config,uv,device)
    out=L03.output_dir(10,args.sequence,args.run)/args.experiment/f"{args.checkpoint[:-3]}_{args.split}"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("평가 출력이 이미 있다. 기존 report를 보관하고 새 평가를 실행하세요.")
    out.mkdir(parents=True,exist_ok=True)
    count=scene["manifest"]["num_frames"]
    ids=list(range(count)) if args.split=="all" else scene["split"][args.split]
    membership={i:name for name in ("train","validation","gap") for i in scene["split"][name]}
    rows=[]
    background=torch.ones(3,device=device)
    previous=None
    with torch.no_grad():
        for slot,index in enumerate(ids):
            target=dataset.batch([index],device)
            gaussians=L03.lesson(8).forward_frame_batch(model,surface,scene,[index],device)
            rendered=L03.lesson(8).render_batch(gaussians,dataset,device,background)[0]
            expected=L03.lesson(7).composite(target["rgb"],target["alpha"],background)[0]
            prediction=rendered["rgb"]
            row=L03.lesson(8).image_metrics(prediction,expected,target["alpha"][0],loss)
            a,b=rendered["alpha"]>.5,target["alpha"][0]>.5
            row.update(frame_id=index,split=membership[index],
                time_seconds=scene["manifest"]["frames"][index]["sequence_time_s"],
                alpha_iou=float((a&b).sum()/(a|b).sum().clamp_min(1)),temporal_residual_l1=None,
                mean_displacement=float(gaussians["displacement"].norm(dim=-1).mean()),
                p95_displacement=float(torch.quantile(gaussians["displacement"].norm(dim=-1),.95)),
                visible_gaussians=int((rendered["radii"]>0).sum()))
            if previous is not None and previous[0]==index-1:
                delta=(prediction-previous[1])-(expected-previous[2])
                mask=torch.maximum(target["alpha"][0],previous[3])
                row["temporal_residual_l1"]=float((delta.abs()*mask).sum()/(mask.sum().clamp_min(1)*3))
            previous=(index,prediction,expected,target["alpha"][0])
            rows.append(row)
            if slot%25==0:
                print(f"10 evaluate {slot}/{len(ids)} frame={index} foreground PSNR={row['foreground_psnr']:.3f}",flush=True)
    metric_keys=("psnr","foreground_psnr","ssim","lpips","alpha_iou","temporal_residual_l1","mean_displacement","p95_displacement")
    summary={}
    for name in ("all","train","validation","gap"):
        group=rows if name=="all" else [row for row in rows if row["split"]==name]
        if not group:
            continue
        summary[name]={"frames":len(group)}
        for key in metric_keys:
            values=[row[key] for row in group if row[key] is not None]
            summary[name][key]=sum(values)/len(values) if values else None
    worst=sorted(rows,key=lambda row:row["foreground_psnr"])[:8]
    with torch.no_grad():
        for row in worst:
            index=row["frame_id"]
            target=dataset.batch([index],device)
            gaussians=L03.lesson(8).forward_frame_batch(model,surface,scene,[index],device)
            rendered=L03.lesson(8).render_batch(gaussians,dataset,device,background)[0]
            expected=L03.lesson(7).composite(target["rgb"],target["alpha"],background)[0]
            image=torch.cat((expected,rendered["rgb"],(expected-rendered["rgb"]).abs()*4),-1)
            L03.save_image(out/"worst_frames"/f"{index:06d}_{row['split']}.jpg",image)
    with (out/"frames.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    L03.save_json(out/"metrics.json",{"summary":summary,"worst_frames":worst,"checkpoint_step":checkpoint["step"],
        "scope":"single identity; tracking on full sequence; UV texture uses train RGB only; held-out Gaussian RGB evaluation",
        "selection":"best.pt selected on validation foreground PSNR; validation is not an untouched test set",
        "temporal_metric":"reference-subtracted consecutive-frame RGB residual; no optical-flow alignment"})
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    for axis,key in zip(axes,("foreground_psnr","lpips","temporal_residual_l1")):
        for name,color in (("train","tab:blue"),("validation","tab:orange"),("gap","tab:gray")):
            group=[row for row in rows if row["split"]==name and row[key] is not None]
            axis.scatter([r["time_seconds"] for r in group],[r[key] for r in group],s=8,label=name,color=color)
        axis.set_ylabel(key)
        axis.grid(alpha=.2)
        axis.legend()
    axes[-1].set_xlabel("sequence time (s)")
    fig.tight_layout()
    fig.savefig(out/"metric_curves.png",dpi=150)
    plt.close(fig)
    training_path=L03.lesson(8).experiment_dir(scene,args.experiment)/"training.jsonl"
    if training_path.exists():
        import json
        by_step={}
        for line in training_path.read_text().splitlines():
            if line:
                row=json.loads(line)
                by_step[row["step"]]=row
        records=[by_step[key] for key in sorted(by_step)]
        if records:
            fig,axes=plt.subplots(2,1,figsize=(12,8))
            for key in records[0]["weighted"]:
                axes[0].plot([r["step"] for r in records],[r["weighted"][key] for r in records],label=key)
            axes[0].set_yscale("symlog",linthresh=1e-5)
            axes[0].legend(ncol=3)
            axes[0].set_title("Weighted losses")
            for key in records[0]["gradient_by_module_after_clip"]:
                axes[1].plot([r["step"] for r in records],[r["gradient_by_module_after_clip"][key] for r in records],label=key)
            axes[1].set_yscale("symlog",linthresh=1e-8)
            axes[1].legend()
            axes[1].set_title("Gradient by module")
            fig.tight_layout()
            fig.savefig(out/"learning_curves.png",dpi=150)
            plt.close(fig)
    print("10 완료:",out)
    print(summary)


if __name__=="__main__":
    main()
