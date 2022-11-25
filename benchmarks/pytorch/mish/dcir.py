#!/usr/bin/python3

# Desc: Generates MLIR code for DCIR
# Usage: python3 dcir.py

import torch
import torch_mlir


# Load model
class Mish(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.log(1 + torch.exp(x))
        return x


model = Mish()
model.eval()

# Compile model
data = torch.rand(8, 32, 224, 224)

module = torch_mlir.compile(model,
                            data,
                            output_type=torch_mlir.OutputType.MHLO)

# Output MLIR code
print(module)
