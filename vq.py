import numpy as np
from scipy.ndimage import gaussian_filter1d
import scipy.cluster
import matplotlib.pyplot as plt
import sklearn.decomposition, sklearn.preprocessing
from compress import normalized_compress_len

def gaussian_pyramid(signal, factor=2, min_length=1):
    """Take a MxN signal, and progressively blur
    and decimate the signal until it becomes [min_length x N].
    Return each level of the pyramid, from full resolution to min_length.
        len(signal)xN
        len(signal)/2 x N
        len(signal)/4 x N
        ...
        1 x N
        (assuming factor=2 and min_length=1)
    """
    pyramid_levels = [signal]
    std_dev = np.sqrt(factor**2 - 1)  # Calculate standard deviation for Gaussian filter
    while len(pyramid_levels[-1])>min_length:        
        smoothed_signal = gaussian_filter1d(pyramid_levels[-1], std_dev, axis=0)
        downsampled_signal = smoothed_signal[::factor]
        pyramid_levels.append(downsampled_signal)
    return pyramid_levels


def pca_range(n):
    """Generate a sequence of integers decreasing by a factor of 2,
    but always include 1 and 2 if n>1."""
    r = [n]
    while n>1:        
        if n>3:
            n = n // 2
        else:
            n = n - 1
        r.append(n)
    return r

def vq(m, k, whiten="standard", pca=None):
    """
    Given a MxN matrix m representing a signal with N attributes
    and an integer k, vector quantize m, and return
    M cluster indices and the average distortion 
    (i.e. average distance to cluster centroids over the whole dataset)
    whiten can be: "standard", "sphere" (i.e. covariance), "minmax", or "none"
    if `pca` is not None, then PCA is performed and `pca` dimensions are kept before VQ
    Returns:
        codes: the vector quantized version of m
        distortion: the average distortion
    """
    if whiten=="standard":        
        m_white = sklearn.preprocessing.StandardScaler().fit_transform(m)
    elif whiten=="sphere":
        # PCA with no reduction
        m_white = sklearn.decomposition.PCA(n_components=m.shape[1], whiten=True).fit_transform(m)
    elif whiten=="minmax":
        m_white = sklearn.preprocess.MinMaxScaler().fit_transform(m)
    elif whiten == "none":
        m_white = m
    # allow PCA reduction if requested
    if pca is not None:
        m_white = sklearn.decomposition.PCA(m_white, n_components=pca, whiten=True).fit_transform(m)
    code, distortion = scipy.cluster.vq.kmeans(m_white, k)
    codes, dists = scipy.cluster.vq.vq(m_white, code)
    return codes, np.mean(dists)

def vq_range(m, ks, **kwargs):
    """
    Vector quantize m with each of k clusters from the sequence ks
    return the list of vector quantised version and the avg. distortion for each k.
    Returns:
        all_codes: the vector quantized version of m for each k
        all_dists: the average distortion for each k
    """
    all_codes = []
    all_dists = []
    for k in ks:
        codes, dists = vq(m, k, **kwargs)
        all_dists.append(dists)
        all_codes.append(codes)
    return all_codes, all_dists

def compression_curve(m, ks, n_surrogates = 5, **kwargs):
    """
        Compress m with VQ clusters from the sequence ks
        Any keyword arguments are passed to the vector quantizer.
        Returns:
            z_curve: the compression ratio for each k
            all_dists: the average distortion for each k        
    """        
    all_codes, all_dists = vq_range(m, ks, **kwargs)
    z_curve = [normalized_compress_len(code_seq) for code_seq in all_codes]
    
    return z_curve, all_dists


def compression_surrogate_curve(m, ks, n_surrogates = 5, **kwargs):
    """
        Compress m with VQ clusters from the sequence ks
        and then create n_surrogates shuffled surrogate sets, and compress them.
        Any keyword arguments are passed to the vector quantizer.
        Returns:
            z_curve: the compression ratio for each k
            z_surrogate: the average compression ratio for the shuffled surrogates
            all_dists: the average distortion for each k
    """        
    all_codes, all_dists = vq_range(m, ks, **kwargs)
    z_curve = [normalized_compress_len(code_seq) for code_seq in all_codes]
    z_surrogate = np.mean([[normalized_compress_len(np.random.permutation(code_seq)) for code_seq in all_codes] for i in range(n_surrogates)], axis=0)
    return z_curve, z_surrogate, all_dists


default_cluster_range = np.unique((2.0 ** np.linspace(1,7.99,65)).astype(np.int64))