import numpy as np
import torch

def fit_approx_polynomial(array, min_value, max_value, degree=100, bins='auto'):
    counts, edges = np.histogram(array, bins=bins, range=(min_value, max_value), density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    coeffs = np.polyfit(centers, counts, deg=degree)
    return np.poly1d(coeffs)

def get_y(poly_model, x):

    y = poly_model(x)
    y[y < 0] = 0
    return y

def get_density(tensor, n_values, min_value, max_value):
    bin_width = (max_value - min_value) / n_values
    counts = torch.histc(tensor.float(), bins=n_values, min=min_value, max=max_value)
    density = counts / (counts.sum() * bin_width)

    return density
