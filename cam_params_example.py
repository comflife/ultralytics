import numpy as np

# 예시 카메라 파라미터 (실제 값으로 교체 필요)
wide_K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], dtype=np.float32)  # 3x3
narrow_K = np.array([[1200, 0, 960], [0, 1200, 540], [0, 0, 1]], dtype=np.float32)  # 3x3
wide_P = np.array([[1000, 0, 960, 0], [0, 1000, 540, 0], [0, 0, 1, 0]], dtype=np.float32)  # 3x4
narrow_P = np.array([[1200, 0, 960, 0], [0, 1200, 540, 0], [0, 0, 1, 0]], dtype=np.float32)  # 3x4

np.savez('camera_params.npz', wide_K=wide_K, wide_P=wide_P, narrow_K=narrow_K, narrow_P=narrow_P)
print('Saved camera_params.npz')
