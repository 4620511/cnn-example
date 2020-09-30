import random

import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda


def seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)  # type: ignore
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)  # type: ignore
    torch.backends.cudnn.deterministic = True
