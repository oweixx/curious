# 출처와 구현 범위

- Kimi Team et al., **Attention Residuals**, arXiv:2603.15031, 2026-03-16.
  [논문](https://arxiv.org/abs/2603.15031), [PDF](https://arxiv.org/pdf/2603.15031).
- [공식 README / 핵심 의사코드](https://github.com/MoonshotAI/Attention-Residuals/blob/master/README.md).
  확인일: 2026-11-13. 공식 공개 저장소의 전체 학습 코드를 복사한 것이 아니다.

이 폴더의 코드는 논문 Eq.1–6 및 Algorithm1의 연산을 설명하기 위한 독립 구현이다.
공식 논문의 텍스트/그림/PDF나 제3자의 재구현 파일을 vendoring하지 않았다.

| 부분 | 구현 기준 |
|---|---|
| Full AttnRes | pseudo-query, key RMSNorm, depth softmax, unnormalized value, zero query initialization |
| Block AttnRes | embedding 별도 source, sublayer output의 block 합, partial block, 최종 head mixer |
| Two-phase | inter-block query batching + stable softmax merge, naive 경로와 수치 및 gradient 검증 |
| 비교 backbone / 과제 / optimizer | 직접 작성한 작은 causal Transformer, synthetic key-color recall, AdamW |
| 대규모 시스템 | pipeline/tensor parallel, distributed cache, fused kernel은 구현하지 않음 |
| 성능 범위 | 작은 과제의 실제 학습·추론. 논문의 corpus, Kimi Linear/MoE, scaling law 재현이 아님 |

기본 attention의 출발점은 Vaswani et al.,
[Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017이다.
