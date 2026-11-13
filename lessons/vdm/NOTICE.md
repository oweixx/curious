# Sources and reproduction boundary

Paper: Simon Giebenhain et al., Video Diffusion Avatars: Taming Video Diffusion Models into Controllable 3D Avatars, arXiv:2610.23772v1.
https://arxiv.org/html/2610.23772v1

Official source: https://github.com/SimonGiebenhain/video-diffusion-avatars/tree/8ba8b8f3e43c98df2e99fe46906cf011b10d34df

Downloaded source is kept intact in ignored `.cache/upstream`. Its LICENSE and component notices remain there. VDA code/weights are CC BY-NC-SA 4.0; component licences (DiffSynth, X-NeMo, pixel3DMM, WAN, FLAME, MHR and data) remain applicable. Integration code adapting the public VDA architecture is provided under CC BY-NC-SA 4.0 with attribution to the VDA authors. No third-party weights or licensed dataset files are committed.

This is an educational reconstruction of missing training loops, not the authors' training implementation. Exact pretrained architecture is imported; paper equations and table settings are implemented in the numbered lessons. Unpublished choices, paper/release discrepancies, and the unimplemented raw-data MHR fitting/export integration are itemized in GUIDE.md. In addition to module smoke tests, actual pretrained 14B forward/backward, optimizer/checkpoint integration and public-example video generation were tested on physical GPU 4. Training integration uses explicitly synthetic targets and does not establish NeRSemble reconstruction quality, long-run convergence or paper-level reproduction. Detailed evidence is in ignored `.cache/preflight/`; the scope is summarized in GUIDE.md.
