def check_cuda_availability():
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

    return "cpu"