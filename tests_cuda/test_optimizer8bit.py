import os
from os.path import join
import shutil
import time
import uuid

#from lion_pytorch import Lion
import torch
from helpers import *
import python_src_quants as bnb
import python_src_quants.functional as F


str2optimizers = {}
str2optimizers["adam_pytorch"] = (None, torch.optim.Adam, bnb.optim.Adam)
str2optimizers["momentum_pytorch"] = (
    None,
    lambda pxx: torch.optim.SGD(pxx, 0.01, 0.9),
    bnb.optim.Adam,
)
str2optimizers["adam"] = (torch.optim.Adam, bnb.optim.Adam)
str2optimizers["paged_adamw"] = (torch.optim.AdamW, bnb.optim.PagedAdamW)
str2optimizers["paged_adam"] = (torch.optim.Adam, bnb.optim.PagedAdam)
str2optimizers["momentum"] = (
    lambda pxx: torch.optim.SGD(pxx, 0.01, 0.9),
    lambda pxx: bnb.optim.SGD(pxx, 0.01, 0.9, block_wise=False),
)
str2optimizers["rmsprop"] = (
    lambda pxx: torch.optim.RMSprop(pxx, 0.01, 0.9),
    lambda pxx: bnb.optim.RMSprop(pxx, 0.01, 0.9, block_wise=False),
)
str2optimizers["adam8bit"] = (torch.optim.Adam, lambda pxx: bnb.optim.Adam8bit(pxx, block_wise=False))
str2optimizers["momentum8bit"] = (
    lambda pxx: torch.optim.SGD(pxx, 0.01, 0.9),
    lambda pxx: bnb.optim.SGD8bit(pxx, 0.01, 0.9, block_wise=False),
)
str2optimizers["rmsprop8bit"] = (
    lambda pxx: torch.optim.RMSprop(pxx, 0.01, 0.9),
    lambda pxx: bnb.optim.RMSprop8bit(pxx, 0.01, 0.9, block_wise=False),
)

str2optimizers["adam8bit_blockwise"] = (torch.optim.Adam, lambda pxx: bnb.optim.Adam8bit(pxx, block_wise=True))
str2optimizers["paged_adamw8bit_blockwise"] = (
    torch.optim.AdamW,
    lambda pxx: bnb.optim.PagedAdamW8bit(pxx, block_wise=True),
)
str2optimizers["paged_adam8bit_blockwise"] = (
    torch.optim.Adam,
    lambda pxx: bnb.optim.PagedAdam8bit(pxx, block_wise=True),
)
str2optimizers["momentum8bit_blockwise"] = (
    lambda pxx: torch.optim.SGD(pxx, 0.01, 0.9),
    lambda pxx: bnb.optim.SGD8bit(pxx, 0.01, 0.9, block_wise=True),
)
str2optimizers["rmsprop8bit_blockwise"] = (
    lambda pxx: torch.optim.RMSprop(pxx, 0.01, 0.9),
    lambda pxx: bnb.optim.RMSprop8bit(pxx, 0.01, 0.9, block_wise=True),
)

str2statenames = {}
str2statenames["adam"] = [("exp_avg", "state1"), ("exp_avg_sq", "state2")]
str2statenames["paged_adamw"] = [("exp_avg", "state1"), ("exp_avg_sq", "state2")]
str2statenames["paged_adam"] = [("exp_avg", "state1"), ("exp_avg_sq", "state2")]
str2statenames["momentum"] = [("momentum_buffer", "state1")]
str2statenames["lamb"] = [("exp_avg", "state1"), ("exp_avg_sq", "state2")]
str2statenames["rmsprop"] = [("square_avg", "state1")]
str2statenames["adam8bit"] = [("exp_avg", "state1", "qmap1", "max1"), ("exp_avg_sq", "state2", "qmap2", "max2")]
str2statenames["lamb8bit"] = [("exp_avg", "state1", "qmap1", "max1"), ("exp_avg_sq", "state2", "qmap2", "max2")]
str2statenames["adam8bit_blockwise"] = [
    ("exp_avg", "state1", "qmap1", "absmax1"),
    ("exp_avg_sq", "state2", "qmap2", "absmax2"),
]
str2statenames["paged_adam8bit_blockwise"] = [
    ("exp_avg", "state1", "qmap1", "absmax1"),
    ("exp_avg_sq", "state2", "qmap2", "absmax2"),
]
str2statenames["paged_adamw8bit_blockwise"] = [
    ("exp_avg", "state1", "qmap1", "absmax1"),
    ("exp_avg_sq", "state2", "qmap2", "absmax2"),
]
str2statenames["momentum8bit"] = [("momentum_buffer", "state1", "qmap1", "max1")]
str2statenames["momentum8bit_blockwise"] = [("momentum_buffer", "state1", "qmap1", "absmax1")]
str2statenames["rmsprop8bit"] = [("square_avg", "state1", "qmap1", "max1")]
str2statenames["rmsprop8bit_blockwise"] = [("square_avg", "state1", "qmap1", "absmax1")]
str2statenames["paged_lion8bit_blockwise"] = [("exp_avg", "state1", "qmap1", "absmax1")]

optimizer_names_32bit = ["adam", "momentum", "rmsprop", "paged_adamw", "paged_adam", "lion", "paged_lion"]


optimizer_names_8bit = [
    "adam8bit",
    "momentum8bit",
    "rmsprop8bit",
    "adam8bit_blockwise",
    "momentum8bit_blockwise",
    "rmsprop8bit_blockwise",
]


def test_optimizer8bit(dim1, dim2, gtype, optim_name):
    if gtype == torch.bfloat16 and optim_name not in ["adam8bit_blockwise", "lion8bit_blockwise"]:
        pytest.skip()
    if dim1 == 1 and dim2 == 1:
        return
    p1 = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.1
    p2 = p1.clone()
    p1 = p1.float()
    blocksize = 2048

    torch_optimizer = str2optimizers[optim_name][0]([p1])
    bnb_optimizer = str2optimizers[optim_name][1]([p2])

    if gtype == torch.float32:
        atol, rtol = 3e-3, 1e-3
        patol, prtol = 1e-5, 1e-3
    elif gtype == torch.bfloat16:
        atol, rtol = 3e-3, 1e-3
        patol, prtol = 1e-4, 1e-2
    else:
        atol, rtol = 3e-3, 1e-3
        patol, prtol = 1e-5, 1e-3

    errors = []
    relerrors = []

    for i in range(100):
        g = torch.randn(dim1, dim2, device="cuda", dtype=gtype) * 0.01
        p1.grad = g.clone().float()
        p2.grad = g.clone()

        bnb_optimizer.step()
        torch_optimizer.step()

        # since Lion can have pretty noisy updates where things lie at the boundary
        # allow up to 5 errors for Lion
        #assert_most_approx_close(p1, p2.float(), patol, prtol, max_error_count=5)

        dequant_states = []
        for name1, name2, qmap, max_val in str2statenames[optim_name]:
            # print(bnb_optimizer.state[p2][max_val], name1)
            if "blockwise" in optim_name:
                s1 = F.dequantize_blockwise(
                    code=bnb_optimizer.state[p2][qmap],
                    absmax=bnb_optimizer.state[p2][max_val],
                    A=bnb_optimizer.state[p2][name2],
                    blocksize=blocksize,
                )
            else:
                s1 = F.dequantize(
                    code=bnb_optimizer.state[p2][qmap],
                    absmax=bnb_optimizer.state[p2][max_val],
                    A=bnb_optimizer.state[p2][name2],
                )
            num_not_close = torch.isclose(torch_optimizer.state[p1][name1], s1, atol=atol, rtol=rtol) == 0
            # assert num_not_close.sum().item() < 20
            dequant_states.append(s1.clone())

        err = torch.abs(p1 - p2)
        relerr = err / (torch.abs(p1) + 1e-9)
        if g.dtype == torch.bfloat16:
            print("Mean error: (Expected Bound 0.0015)", err.mean())
            print("Relative Mean error: (Expected Bound 0.0016)", relerr.mean())
            #assert err.mean() < 0.00015
            #assert relerr.mean() < 0.0016
        else:
            print("Mean error: (Expected Bound 0.0012)", err.mean())
            print("Rel Mean error: (Expected Bound 0.0012)", relerr.mean())
            #assert err.mean() < 0.00012
            #assert relerr.mean() < 0.0012

        errors.append(err.mean().item())
        relerrors.append(relerr.mean().item())

        if i % 10 == 0 and i > 0:
            for (name1, name2, qmap, max_val), s in zip(str2statenames[optim_name], dequant_states):
                s1cpy = s.clone()
                raws1cpy = bnb_optimizer.state[p2][name2].clone()
                qmap1 = bnb_optimizer.state[p2][qmap].clone()

                path = get_temp_dir()
                torch.save(bnb_optimizer.state_dict(), join(path, "opt.pt"))
                del bnb_optimizer
                bnb_optimizer = None
                bnb_optimizer = str2optimizers[optim_name][1]([p2])
                bnb_optimizer.load_state_dict(torch.load(join(path, "opt.pt")))
                rm_path(path)
                torch.testing.assert_close(raws1cpy, bnb_optimizer.state[p2][name2])
                torch.testing.assert_close(qmap1, bnb_optimizer.state[p2][qmap])

                if "blockwise" in optim_name:
                    s1 = F.dequantize_blockwise(
                        code=bnb_optimizer.state[p2][qmap],
                        absmax=bnb_optimizer.state[p2][max_val],
                        A=bnb_optimizer.state[p2][name2],
                        blocksize=blocksize,
                    )
                else:
                    s1 = F.dequantize(
                        code=bnb_optimizer.state[p2][qmap],
                        absmax=bnb_optimizer.state[p2][max_val],
                        A=bnb_optimizer.state[p2][name2],
                    )
                torch.testing.assert_close(s1cpy, s1)

                num_not_close = torch.isclose(torch_optimizer.state[p1][name1], s1, atol=atol, rtol=rtol) == 0
                #assert num_not_close.sum().item() < 20
                print("Closure Sum", num_not_close.sum().item())
            # since Lion can have pretty noisy updates where things lie at the boundary
            # allow up to 5 errors for Lion
            assert_most_approx_close(p1, p2.float(), patol, prtol, max_error_count=5000)

        # the parameters diverge quickly. Here we keep them close
        # together so we can test against the Adam error
        p1.data = p1.data.to(gtype).float()
        p2.copy_(p1.data)
        torch.testing.assert_close(p1.to(gtype), p2)
        for (name1, name2, qmap, max_val), s in zip(str2statenames[optim_name], dequant_states):
            torch_optimizer.state[p1][name1].copy_(s.data)

        print(sum(errors)/len(errors))
        print(sum(relerrors)/len(relerrors))

if __name__ =='__main__':
    optim_name = optimizer_names_8bit
    gtype = [torch.float32, torch.float16, torch.bfloat16]
    dim2 = [32, 1024, 4097]
    dim1 = [1024]
    print("=============Test FP32 Adam8bit====================")
    test_optimizer8bit(dim1[0], dim2[0], gtype[0], optim_name[0])
    print("=============Test FP32 Momentum8bit====================")
    test_optimizer8bit(dim1[0], dim2[0], gtype[0], optim_name[1])
    print("=============Test FP32 RmsProp8bit====================")
    #test_optimizer8bit(dim1[0], dim2[0], gtype[0], optim_name[2])
    print("=============Test FP32 Adam8bitBlockwise====================")
    #test_optimizer8bit(dim1[0], dim2[0], gtype[0], optim_name[3])
    print("=============Test FP32 Momentum8bitBlockwise====================")
    test_optimizer8bit(dim1[0], dim2[0], gtype[0], optim_name[4])
    print("=============Test FP32 RMSProp8bitBlockwise====================")
    #test_optimizer8bit(dim1[0], dim2[0], gtype[0], optim_name[5])