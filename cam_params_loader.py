import numpy as np

def load_camera_params(npz_path='camera_params.npz'):
    params = np.load(npz_path, allow_pickle=True)
    wide_K = params['wide_K']
    wide_P = params['wide_P']
    narrow_K = params['narrow_K']
    narrow_P = params['narrow_P']
    return wide_K, wide_P, narrow_K, narrow_P

if __name__ == '__main__':
    wide_K, wide_P, narrow_K, narrow_P = load_camera_params()
    print('wide_K:\n', wide_K)
    print('wide_P:\n', wide_P)
    print('narrow_K:\n', narrow_K)
    print('narrow_P:\n', narrow_P)
