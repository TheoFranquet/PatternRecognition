from scipy.io import loadmat
import numpy as np
from sklearn.preprocessing import scale

def load_face_data(filename, normalise = False):
    data = loadmat(filename)
    faces = data['X'].T
    labels = data['l'][0]

    if normalise:
        faces = scale(faces.astype(float))
    return faces, labels


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

if __name__ == '__main__':
    print(load_face_data('face.mat')[0].shape)