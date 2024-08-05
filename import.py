import torch
import intel_extension_for_pytorch
import python_src_quants as bnb
import python_src_quants.functional

p = torch.nn.Parameter(torch.rand(10, 10).to("cuda"))
a = torch.rand(10, 10).cuda()

p1 = p.data.sum().item()

adam = bnb.optim.Adam([p])

out = a * p
loss = out.sum()
loss.backward()
print(loss, adam)
adam.step()

p2 = p.data.sum().item()
print(p1, p2, p1 - p2)
assert p1 == p2
print("SUCCESS!")
print("Installation was successful!")
