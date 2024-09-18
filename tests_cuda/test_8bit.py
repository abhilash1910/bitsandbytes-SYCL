import torch
import torch.nn as nn

import python_src_quants as bnb
from python_src_quants.nn import Linear8bitLt, Int8Params

def test_linear_no_igemmlt():
    linear = torch.nn.Linear(1024, 3072)
    x = torch.randn(3, 1024, dtype=torch.half)
    linear_custom = Linear8bitLt(
        linear.in_features,
        linear.out_features,
        linear.bias is not None,
        has_fp16_weights=False,
        threshold=6.0,
    )
    linear_custom.state.force_no_igemmlt = False

    linear_custom.weight = Int8Params(
        linear.weight.data.clone(), requires_grad=False, has_fp16_weights=True
    ).to(linear.weight.dtype)
    linear_custom.bias = linear.bias
    linear_custom = linear_custom.cuda()
    linear = linear.half().cuda()
    print("Completed linear layer")
    x_ref = x.clone().cuda().requires_grad_(True)
    x_ours = x.clone().cuda().requires_grad_(True)
    fx_ref = linear(x_ref).float()
    grad_proj = torch.randn_like(fx_ref)
    (fx_ref * grad_proj).mean().backward()
    print("After backward")
    print(linear_custom.state)
    fx_ours = linear_custom(x_ours).float()
    (fx_ours * grad_proj).mean().backward()
    print("After backward")
    assert torch.allclose(fx_ref, fx_ours, atol=0.02)
    assert torch.allclose(x_ref.grad, x_ours.grad, atol=0.01)
    assert not linear_custom.state.has_fp16_weights
    assert linear_custom.state.CB is not None
    assert linear_custom.state.CxB is None

test_linear_no_igemmlt()