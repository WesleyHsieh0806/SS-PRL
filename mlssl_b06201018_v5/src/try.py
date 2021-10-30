import torch
grid_per_side = [3, 5]
cum_nofgrid = torch.cumsum(torch.Tensor([2*(nof_grid**2)
                                         for nof_grid in grid_per_side]), 0)
local_offset = 8
print([(local_offset+cum_nofgrid[i-1], local_offset+cum_nofgrid[i]) if i!=0 else (local_offset, local_offset+cum_nofgrid[i]) for i in range(len(grid_per_side))])
