# salida de corsario → tensor para ML
# geometría → vectores

import torch

def convert(np_data):

    torch_data = torch.from_numpy(np_data, dtype=torch.float32)

    return torch_data