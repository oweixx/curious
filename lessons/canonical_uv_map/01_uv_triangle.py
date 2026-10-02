"""첫 질문: UV 주소 하나가 어떻게 3D 위치 하나를 가리킬까?

위에서부터 계산을 읽고, 실행 전에 각 print의 값을 예상해보자.
이 파일의 삼각형은 직접 정한 작은 예제다. 다음에 FLAME의 실제 삼각형으로 바꾼다.
"""

import numpy as np

# 1. 같은 삼각형의 세 꼭짓점에 두 종류의 좌표를 붙인다.
# vertices_3d[i]와 vertices_uv[i]는 같은 꼭짓점을 뜻한다.
# UV는 3D에서 카메라로 투영해 얻은 값이 아니라, 따로 부여한 표면 주소다.
vertices_3d = np.array([
    [0.0, 0.0, 0.0],  # A: 실제 공간의 위치
    [2.0, 0.0, 0.0],  # B
    [0.0, 1.0, 1.0],  # C: Z가 1이므로 XY 평면에 놓인 삼각형이 아니다.
])
vertices_uv = np.array([
    [0.0, 0.0],  # A의 UV 주소
    [1.0, 0.0],  # B의 UV 주소
    [0.0, 1.0],  # C의 UV 주소
])

# 2. 삼각형 내부의 UV 주소 하나를 정한다. 아직 이미지나 픽셀 격자는 없다.
point_uv = np.array([0.25, 0.25])
a_uv, b_uv, c_uv = vertices_uv
edge_ab = b_uv - a_uv
edge_ac = c_uv - a_uv
from_a = point_uv - a_uv

# 3. point_uv = A_uv + weight_b * AB_uv + weight_c * AC_uv 를 푼다.
# u와 v에 대해 식 두 개, 미지수 두 개가 생긴다.
# 아래는 그 2x2 연립방정식을 직접 계산한 식이다.
determinant = edge_ab[0] * edge_ac[1] - edge_ab[1] * edge_ac[0]
if abs(determinant) < 1e-12:
    raise ValueError('UV 삼각형이 선/점으로 납작해져 내부 좌표를 계산할 수 없다.')
weight_b = (from_a[0] * edge_ac[1] - from_a[1] * edge_ac[0]) / determinant
weight_c = (edge_ab[0] * from_a[1] - edge_ab[1] * from_a[0]) / determinant
weight_a = 1.0 - weight_b - weight_c
print('UV 주소:', point_uv)
print('A, B, C의 가중치:', weight_a, weight_b, weight_c)

# 가중치의 합은 1. 모두 0 이상이면 삼각형 내부/경계다.
inside = min(weight_a, weight_b, weight_c) >= -1e-12
print('삼각형 안인가?', inside)

# 4. 같은 가중치를 3D 꼭짓점에 적용한다. 이것이 barycentric interpolation이다.
a_3d, b_3d, c_3d = vertices_3d
point_3d = weight_a * a_3d + weight_b * b_3d + weight_c * c_3d
print('같은 가중치로 복원한 UV:', weight_a * a_uv + weight_b * b_uv + weight_c * c_uv)
print('대응하는 3D 위치:', point_3d)

# 직접 바꿔볼 것:
# - point_uv를 [0, 0], [1, 0], [0, 1]로 바꾸면 각각 어느 3D 꼭짓점이 나올까?
# - point_uv를 [0.75, 0.75]로 바꾸면 왜 가중치 하나가 음수가 될까?
#   이때 계산되는 point_3d는 삼각형 밖으로 외삽한 위치라 표면 sample로 쓰지 않는다.
# - C의 Z만 1에서 2로 바꾸면 UV 가중치와 3D 결과 중 무엇이 달라질까?
#
# 다음 단계: UV 픽셀마다 이 계산을 반복해 [H, W, 3] position map을 만든다.
# 그 뒤 이 예제의 꼭짓점들을 FLAME의 vertex/UV 데이터로 교체한다.
