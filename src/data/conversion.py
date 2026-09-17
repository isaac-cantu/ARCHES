# salida de corsario → tensor para ML
# geometría → vectores

import numpy as np
import torch


def convert(np_data):
    """Convert a NumPy array to a float32 torch.Tensor.

    Note: torch.from_numpy() does not accept a dtype= argument (it always
    shares the array's existing dtype); the previous version of this
    function (`torch.from_numpy(np_data, dtype=torch.float32)`) raised
    TypeError on every call. Currently unused by run_pipeline.py (the
    active data path is data_processed(), which returns torch tensors
    directly), but fixed here so it works if/when it is wired back in.
    """
    np_data = np.asarray(np_data, dtype=np.float32)
    return torch.from_numpy(np_data)