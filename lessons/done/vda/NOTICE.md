# Sources and reproduction boundary

Paper: Simon Giebenhain et al., Video Diffusion Avatars: Taming Video Diffusion Models into Controllable 3D Avatars, arXiv:2610.23772v1.
https://arxiv.org/html/2610.23772v1

Official source: https://github.com/SimonGiebenhain/video-diffusion-avatars/tree/8ba8b8f3e43c98df2e99fe46906cf011b10d34df

Downloaded source is kept intact in ignored `.cache/upstream`. Its LICENSE and component notices remain there. VDA code/weights are CC BY-NC-SA 4.0; component licences (DiffSynth, X-NeMo, pixel3DMM, WAN, FLAME, MHR and data) remain applicable. Integration code adapting the public VDA architecture is provided under CC BY-NC-SA 4.0 with attribution to the VDA authors. No third-party weights or licensed dataset files are committed.

This is an educational reconstruction of missing training loops, not the authors' training implementation. Exact pretrained architecture is imported; paper equations and table settings are implemented in the numbered lessons. Unpublished choices and paper/release discrepancies are itemized in GUIDE.md. The raw-data adapter in `preprocess.py` uses official pixel3DMM, X-NeMo, SAM3D/MHR and proxy rendering; multiview alignment, correspondence choices and the joint fitting/export loop are lesson implementations. SAM source and its license remain in ignored `.cache/sam-3d-body`; model access is separately gated. In addition to module smoke tests, actual pretrained 14B forward/backward, optimizer/checkpoint integration and public-example video generation were tested on physical GPU 4. Training integration uses explicitly synthetic targets and does not establish NeRSemble reconstruction quality, long-run convergence or paper-level reproduction. Detailed evidence is in ignored `.cache/preflight/`; preprocessing checks are kept separately in `.cache/preprocess_smoke/`.

`preprocess.py::stable_tracker_projection` adapts pixel3DMM's `project_points_screen_space` with a signed camera-depth floor to prevent NaN gradients from excluded vertices at z=0. A child-process integration installs this numerical fix and checks optimizer gradients without editing the downloaded source or changing its loss weights/iteration settings. Failure-state replay and regression evidence are kept in ignored `.cache/val_nan_diagnosis/`.

Lesson 01 explicitly supports a `single-subject` experiment on subject 042: the released MLP architecture and four training losses are retained, while contrastive batch sizes use the three available synchronized views and validation holds out time intervals instead of identities. The separate `paper` protocol retains the original batch settings and identity-disjoint validation. Single-subject results do not establish identity generalization.
