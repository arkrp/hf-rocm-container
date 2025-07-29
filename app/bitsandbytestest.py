print('PROGRAM STARTING.. loading libraries')
import torch
import bitsandbytes as bnb
print('libraries loaded! Testing bitsandbytes!')

# Create a dummy tensor and optimizer
param = torch.randn(10, 10, device='cuda', requires_grad=True)
optimizer = bnb.optim.Adam8bit([param])

# Perform a dummy update
loss = param.mean()
loss.backward()
optimizer.step()
optimizer.zero_grad()

print("bitsandbytes functionality test successful.")
