def check_cuda_availability():
    """
    Checks the availability of CUDA for GPU processing.

    This function tries to determine whether CUDA is available using either
    the PyTorch or TensorFlow libraries. It returns the appropriate device 
    type ("cuda" or "cpu") based on the availability.

    Returns:
        str: "cuda" if a CUDA-enabled GPU is available, otherwise "cpu".

    Note:
        If neither PyTorch nor TensorFlow is installed, the function will print
        an error message and fall back to using "cpu".
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    except ImportError:
        print("Torch not available, trying with TensorFlow or install it using 'pip install torch'.")

    try:
        import tensorflow
        if tensorflow.test.is_built_with_cuda():
            return "cuda"
        else:
            return "cpu"
    except ImportError:
        print("TensorFlow not available, install it using 'pip install tensorflow'.")

    return "CPU"