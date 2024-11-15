from typing import Union, Tuple
from scipy.ndimage import convolve1d
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from scipy.signal import find_peaks

import numpy as np

def smooth_columns(matrix: Union[np.ndarray, list], window_size: int) -> np.ndarray:
    """
        Smooths each column of a 2D matrix by applying a 1D moving average filter along the column dimension.

        Parameters:
        ----------
        matrix : np.ndarray
            A 2D NumPy array to be smoothed across columns.
        window_size : int
            The size of the smoothing window, by default 5.

        Returns:
        -------
        np.ndarray
            A 2D NumPy array with smoothed columns, same shape as the input matrix.

        Raises:
        ------
        TypeError
            If `matrix` is not a 2D NumPy array or `window_size` is not an integer.
        ValueError
            If `window_size` is less than 1.

        Example:
        -------
        >>> matrix = np.random.rand(10, 10)
        >>> smoothed_matrix = smooth_columns(matrix, window_size=5)
    """

    # Type and value checks
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input matrix must be a NumPy array.")
    if matrix.ndim != 2:
        raise TypeError("Input matrix must be a 2D NumPy array.")
    if not isinstance(window_size, int):
        raise TypeError("Window size must be an integer.")
    if window_size < 1:
        raise ValueError("Window size must be at least 1.")

    # Create a simple 1D averaging kernel of the desired window size
    kernel = np.ones(window_size) / window_size
    
    # Apply the 1D convolution along the columns (axis=0)
    smoothed_matrix = convolve1d(matrix, kernel, axis=0, mode='nearest')
    
    return smoothed_matrix

def reduce_dim(
    matrix: Union[np.ndarray, list],
    window_size: int,
    **kwargs
) -> np.ndarray:
    """
        Reduces the dimensionality of a given matrix using the t-SNE algorithm, with optional smoothing.

        Parameters:
        ----------
        matrix : np.ndarray 
            A 2D matrix to be reduced in dimensionality.
        window_size : int 
            The window size to use for smoothing the columns of the matrix.
        **kwargs : 
            Arbitrary keyword arguments to be passed directly to the TSNE initializer.

        Returns:
        -------
        np.ndarray:
            The 2D coordinates of the matrix after applying t-SNE.

        Raises:
        ------
        TypeError: 
            If the matrix is not a 2D NumPy array or if it's not an array at all.
        ValueError: 
            If the window size is non-positive or larger than the number of columns in the matrix.

        Example:
        -------
        >>> matrix = np.random.rand(10, 10)

        # `perplexity` is an example of a **kwargs argument passed to TSNE.
        >>> smoothed_matrix = reduce_dim(matrix, window_size = 5, perplexity = 30.0)
    """
    
    # Type and value checks
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input matrix must be a NumPy array.")
    if matrix.ndim != 2:
        raise TypeError("Input matrix must be a 2D NumPy array.")
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("Window size must be a positive integer.")

    # Smooth the data
    smoothed_data = smooth_columns(matrix, window_size)

    # Initialize TSNE with the provided extra arguments and keyword arguments
    tsne = TSNE(**kwargs)

    # Apply tSNE to the smoothed data
    tsne_output = tsne.fit_transform(smoothed_data)

    return tsne_output

def calculate_jumps(data: Union[np.ndarray, list], length_jumps: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
        Calculates Mahalanobis distances between consecutive points in a dataset
        and identifies significant jumps based on these distances. It then calculates 
        the mentation rate in seconds/jump or timepoints/jump.

        Parameters:
        ----------
        data : np.ndarray
            An n x 2 matrix where each row represents a point.
        length_jumps : int (default = 0)
            An integer describing the amount of time inbetween each point. 
            Zero means no rate provided, result will be timepoints/jump vs. seconds/jump.
        
        Returns:
        -------
        Tuple : [np.ndarray, np.ndarray] 
            Indices of the peaks and properties of these peaks.

        Raises:
        ------
        ValueError: 
            If the input data cannot be converted into a NumPy array or is not exactly two-dimensional.

        Example:
        -------
        >>> matrix = np.random.rand(10, 2)
        >>> jumps, indices = calculate_jumps(matrix)
    """
    
    # Ensure the input is a NumPy array of proper shape
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("Input data must be an n x 2 matrix.")
    if not isinstance(length_jumps, int) or length_jumps < 0:
        raise ValueError("Window size must be a positive integer or zero.")

    # Calculate Mahalanobis distances between consecutive points
    try:
        distances = np.diagonal(squareform(pdist(data, metric='mahalanobis')), 1)
    except np.linalg.LinAlgError:
        raise ValueError("Error in computing Mahalanobis distances. Check the input data validity.")

    # Identify significant jumps in the distance metrics
    peaks, _ = find_peaks(distances, prominence = 1)

    # Calculate mentation rate
    timepoints = data.shape[0]
    n_jumps = len(peaks)

    if length_jumps == 0:
        rate = timepoints / n_jumps
    else:
        rate = (length_jumps * timepoints) / n_jumps



    return peaks, rate


    