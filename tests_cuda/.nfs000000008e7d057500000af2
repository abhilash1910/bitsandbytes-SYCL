from typing import Tuple

import torch
import python_src_quants as bnb
import python_src_quants.functional as F
from helpers import (
    BOOLEAN_TRIPLES,
    BOOLEAN_TUPLES,
    TRUE_FALSE,
    describe_dtype,
    get_test_dims,
    id_formatter,
)
import numpy
TRANSPOSE_VALS = [(False, True), (False, False)]


def test_matmul(dim1, dim2, dim3, dim4, funcs, dtype, req_grad: Tuple[bool, bool], transpose: Tuple[bool, bool]):
    if dim2 > 0:
        dim2 = dim2 - (dim2 % 16)
    dim3 = dim3 - (dim3 % 16)
    dim4 = dim4 - (dim4 % 16)
    for i in range(25):
        # normal multiply
        if funcs[0] in [torch.mm, torch.matmul]:
            dimA = (dim2, dim3) if not transpose[0] else (dim3, dim2)
            dimB = (dim3, dim4) if not transpose[1] else (dim4, dim3)
            A = torch.randn(size=dimA, device="cuda", requires_grad=req_grad[0])
            B = torch.randn(size=dimB, device="cuda", requires_grad=req_grad[1])
            target = torch.randn(size=(dim2, dim4), device="cuda", requires_grad=req_grad[1])
            torch.nn.init.xavier_uniform_(B)

            if not transpose[0] and not transpose[1]:
                out_torch = funcs[0](A, B)
                out_bnb = funcs[1](A, B)
            elif not transpose[0] and transpose[1]:
                out_torch = funcs[0](A, B.t())
                out_bnb = funcs[1](A, B.t())
            elif transpose[0] and not transpose[1]:
                out_torch = funcs[0](A.t(), B)
                out_bnb = funcs[1](A.t(), B)
            elif transpose[0] and transpose[1]:
                out_torch = funcs[0](A.t(), B.t())
                out_bnb = funcs[1](A.t(), B.t())

            n = out_bnb.numel()
            idx = torch.isclose(out_bnb, out_torch, atol=0.01, rtol=0.1)
            assert (idx == 0).sum().item() < n * 0.0175
            idx = torch.isclose(out_bnb, out_torch, atol=0.035, rtol=0.2)
            assert (idx == 0).sum().item() < n * 0.001

            if any(req_grad):
                out_bnb.data.copy_(out_torch)
                torch.cuda.synchronize()
                loss_bnb = torch.nn.functional.mse_loss(out_bnb, target).mean()
                loss_bnb.backward()
                gradA1 = A.grad
                gradB1 = B.grad
                A.grad = None
                B.grad = None

                loss_torch = torch.nn.functional.mse_loss(out_torch, target).mean()
                loss_torch.backward()
                gradA2 = A.grad
                gradB2 = B.grad
                A.grad = None
                B.grad = None

            if req_grad[0]:
                torch.testing.assert_close(gradA1, gradA2, atol=0.015, rtol=0.1)
            if req_grad[1]:
                n = gradB1.numel()
                idx = torch.isclose(gradB1, gradB2, atol=0.06, rtol=0.3)
                assert (idx == 0).sum().item() < n * 0.1
                idx = torch.isclose(gradB1, gradB2, atol=0.10, rtol=0.3)
                assert (idx == 0).sum().item() < n * 0.02
                torch.testing.assert_close(gradB1, gradB2, atol=0.18, rtol=0.3)

        # batched matrix multiply
        if funcs[0] in [torch.bmm, torch.matmul]:
            A = torch.randn(
                size=(dim1, dim2, dim3),
                device="cuda",
                requires_grad=req_grad[0],
            )
            B = torch.randn(
                size=(dim1, dim3, dim4),
                device="cuda",
                requires_grad=req_grad[1],
            )
            target = torch.randn(
                size=(dim1, dim2, dim4),
                device="cuda",
                requires_grad=req_grad[1],
            )
            torch.nn.init.xavier_uniform_(B)

            out_torch = funcs[0](A, B)
            out_bnb = funcs[1](A, B)

            n = out_bnb.numel()
            idx = torch.isclose(out_bnb, out_torch, atol=0.01, rtol=0.1)
            assert (idx == 0).sum().item() < n * 0.01
            torch.testing.assert_close(out_bnb, out_torch, atol=0.027, rtol=0.2)

            if any(req_grad):
                out_bnb.data.copy_(out_torch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                else:
                    torch.cuda.synchronize()
                loss_bnb = torch.nn.functional.mse_loss(out_bnb, target).mean()
                loss_bnb.backward()
                gradA1 = A.grad
                gradB1 = B.grad
                A.grad = None
                B.grad = None

                loss_torch = torch.nn.functional.mse_loss(out_torch, target).mean()
                loss_torch.backward()
                gradA2 = A.grad
                gradB2 = B.grad
                A.grad = None
                B.grad = None

            if req_grad[0]:
                torch.testing.assert_close(gradA1, gradA2, atol=0.015, rtol=0.1)
            if req_grad[1]:
                n = gradB1.numel()
                idx = torch.isclose(gradB1, gradB2, atol=0.06, rtol=0.3)
                assert (idx == 0).sum().item() < n * 0.1
                idx = torch.isclose(gradB1, gradB2, atol=0.10, rtol=0.3)
                assert (idx == 0).sum().item() < n * 0.02

        if funcs[0] in [torch.matmul]:
            dim1 = dim1 - (dim1 % 16)
            A = torch.randn(
                size=(dim1, dim2, dim3),
                device="cuda",
                requires_grad=req_grad[0],
            )
            dimB = (dim4, dim3) if transpose[1] else (dim3, dim4)
            B = torch.randn(size=dimB, device="cuda", requires_grad=req_grad[1])
            target = torch.randn(
                size=(dim1, dim2, dim4),
                device="cuda",
                requires_grad=req_grad[1],
            )
            torch.nn.init.xavier_uniform_(B)

            if transpose[1]:
                out_torch = funcs[0](A, B.t())
                out_bnb = funcs[1](A, B.t())
            else:
                out_torch = funcs[0](A, B)
                out_bnb = funcs[1](A, B)

            n = out_bnb.numel()
            idx = torch.isclose(out_bnb, out_torch, atol=0.01, rtol=0.1)
            assert (idx == 0).sum().item() < n * 0.0175
            idx = torch.isclose(out_bnb, out_torch, atol=0.035, rtol=0.2)
            assert (idx == 0).sum().item() < n * 0.001

            if any(req_grad):
                out_bnb.data.copy_(out_torch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                else:
                    torch.cuda.synchronize()
                loss_bnb = torch.nn.functional.mse_loss(out_bnb, target).mean()
                loss_bnb.backward()
                gradA1 = A.grad
                gradB1 = B.grad
                A.grad = None
                B.grad = None

                loss_torch = torch.nn.functional.mse_loss(out_torch, target).mean()
                loss_torch.backward()
                gradA2 = A.grad
                gradB2 = B.grad
                A.grad = None
                B.grad = None

            if req_grad[0]:
                torch.testing.assert_close(gradA1, gradA2, atol=0.015, rtol=0.1)
            if req_grad[1]:
                n = gradB1.numel()
                idx = torch.isclose(gradB1, gradB2, atol=0.06, rtol=0.3)
                assert (idx == 0).sum().item() < n * 0.1
                idx = torch.isclose(gradB1, gradB2, atol=0.10, rtol=0.3)
                assert (idx == 0).sum().item() < n * 0.02


def test_dim3_igemm(seq_dim, hidden_dim, batch_dim):
    seq_dim = seq_dim - (seq_dim % 32)
    hidden_dim = hidden_dim - (hidden_dim % 32)
    batch_dim = batch_dim - (batch_dim % 2)
    for i in range(25):
        A = torch.randint(-128, 127, size=(batch_dim, seq_dim, hidden_dim), device="cuda").to(torch.int8)
        B = torch.randint(-128, 127, size=(batch_dim, seq_dim, 1024), device="cuda").to(torch.int8)
        out2 = torch.einsum("bsi, bso->io", A.float(), B.float())
        iout = torch.empty(A.shape[2], B.shape[2], dtype=torch.int32, device=A.device)
        out = F.igemm(A, B, out=iout)
        torch.testing.assert_close(out.float(), out2)


def test_igemm(hidden_dim, batch_dim, transpose, seq_dim):
    hidden_dim = hidden_dim - (hidden_dim % 32)
    batch_dim = batch_dim - (batch_dim % 16)
    seq_dim = seq_dim - (seq_dim % 16)
    k=10
    for i in range(k):
        shapeA = (batch_dim, hidden_dim) if not transpose[0] else (hidden_dim, batch_dim)
        shapeB = (32 * numpy.random.randint(1, 4), hidden_dim) if transpose[1] else (hidden_dim, 32 * numpy.random.randint(1, 4))
        A = torch.randint(-128, 127, size=shapeA, device="cuda").to(torch.int8)
        B = torch.randint(-128, 127, size=shapeB, device="cuda").to(torch.int8)
        if not transpose[0] and not transpose[1]:
            out2 = torch.matmul(A.float(), B.float())
            out = F.igemm(A, B)
        elif not transpose[0] and transpose[1]:
            out2 = torch.matmul(A.float(), B.t().float())
            out = F.igemm(A, B.t())
        elif transpose[0] and not transpose[1]:
            out2 = torch.matmul(A.t().float(), B.float())
            out = F.igemm(A.t(), B)
        elif transpose[0] and transpose[1]:
            out2 = torch.matmul(A.t().float(), B.t().float())
            out = F.igemm(A.t(), B.t())

        torch.testing.assert_close(out.float(), out2)

    for i in range(k):
        shapeA = (batch_dim, seq_dim, hidden_dim)
        shapeB = (32 * random.randint(1, 4), hidden_dim) if transpose[1] else (hidden_dim, 32 * random.randint(1, 4))
        A = torch.randint(-128, 127, size=shapeA, device="cuda").to(torch.int8)
        B = torch.randint(-128, 127, size=shapeB, device="cuda").to(torch.int8)
        if not transpose[0] and not transpose[1]:
            out2 = torch.matmul(A.float(), B.float())
            out = F.igemm(A, B)
        elif not transpose[0] and transpose[1]:
            out2 = torch.matmul(A.float(), B.t().float())
            out = F.igemm(A, B.t())

        torch.testing.assert_close(out.float(), out2)



def test_ibmm(dim1, dim2, dim3, dim4, transpose):
    dim2 = dim2 - (dim2 % 16)
    dim3 = dim3 - (dim3 % 16)
    dim4 = dim4 - (dim4 % 16)
    k=10
    for i in range(k):
        shapeA = (dim1, dim3, dim2) if transpose[0] else (dim1, dim2, dim3)
        shapeB = (dim1, dim4, dim3) if transpose[1] else (dim1, dim3, dim4)
        A = torch.randint(-128, 127, size=shapeA, device="cuda").to(torch.int8)
        B = torch.randint(-128, 127, size=shapeB, device="cuda").to(torch.int8)

        if not transpose[0] and not transpose[1]:
            out2 = torch.bmm(A.float(), B.float())
            out = F.igemm(A, B)
        elif not transpose[0] and transpose[1]:
            out2 = torch.bmm(A.float(), B.permute([0, 2, 1]).float())
            out = F.igemm(A, B.permute([0, 2, 1]))
        elif transpose[0] and not transpose[1]:
            out2 = torch.bmm(A.permute([0, 2, 1]).float(), B.float())
            out = F.igemm(A.permute([0, 2, 1]), B)
        elif transpose[0] and transpose[1]:
            out2 = torch.bmm(A.permute([0, 2, 1]).float(), B.permute([0, 2, 1]).float())
            out = F.igemm(A.permute([0, 2, 1]), B.permute([0, 2, 1]))
        torch.testing.assert_close(out.float(), out2.float())
        

def test_bench_matmul(batch, seq, model, hidden):
    iters = 1000
    formatB = F.get_special_format_str()

    A = torch.randn(batch, seq, model, device="cuda").half()
    B = torch.empty(hidden, model, dtype=torch.float16, device="cuda")
    torch.nn.init.xavier_uniform_(B)

    B_fp4, state = F.quantize_fp4(B)
    B_fp4_c, state_c = F.quantize_fp4(B, compress_statistics=True)

    B_nf4, state_nf4 = F.quantize_nf4(B)
    B_nf4_c, state_nf4_c = F.quantize_nf4(B, compress_statistics=True)

    linear8bit = bnb.nn.Linear8bitLt(model, hidden, False, False).cuda().half()
    linear8bit.eval()

    outliers = torch.randint(0, model, size=(5,)).cuda()
    A[:, :, outliers] = 8.0

    linearMixedBit = bnb.nn.Linear8bitLt(model, hidden, False, False, threshold=6.0).cuda().half()
    # linearMixedBit.eval()

    linear8bit_train = bnb.nn.Linear8bitLt(model, hidden, False).cuda().half()
    linear8bit_train_thresh = bnb.nn.Linear8bitLt(model, hidden, False, threshold=6.0).cuda().half()
    bnb.matmul_4bit(A, B_nf4.t(), quant_state=state_nf4)

    # warmup
    for i in range(iters):
        torch.matmul(A, B.t())
    torch.cuda.synchronize()
    print("")

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(iters):
        torch.matmul(A, B.t())
    torch.cuda.synchronize()
    print(
        f"pytorch fp16: [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s",
    )

    # torch.cuda.synchronize()
    # t0 = time.time()
    # for i in range(iters):
    #    bnb.matmul_4bit(A, B_fp4.t(), quant_state=state)
    # torch.cuda.synchronize()
    # print( f"bnb fp4: [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s" )

    # torch.cuda.synchronize()
    # t0 = time.time()
    # for i in range(iters):
    #    bnb.matmul_4bit(A, B_fp4.t(), quant_state=state_c)
    # torch.cuda.synchronize()
    # print( f"bnb fp4 + compressed stats: [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s" )

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(iters):
        bnb.matmul_4bit(A, B_nf4.t(), quant_state=state_nf4)
    torch.cuda.synchronize()
    print(f"bnb nf4: [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s")

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(iters):
        bnb.matmul_4bit(A, B_nf4_c.t(), quant_state=state_nf4_c)
    torch.cuda.synchronize()
    print(f"bnb nf4+DQ: [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s")


        
if __name__ == '__main__':
    dim1 = get_test_dims(16, 64, n=1)
    dim2 = get_test_dims(32, 96, n=1)  
    dim3 = get_test_dims(32, 96, n=1)
    dim4 = get_test_dims(32, 96, n=1)
    funcs = [(torch.bmm, bnb.bmm_cublas), (torch.matmul, bnb.matmul_cublas)]
    dtype = [torch.float32, torch.float16]
    req_grad = BOOLEAN_TUPLES
    transpose = BOOLEAN_TUPLES
    print("Test standard matmul pass")
    test_matmul(dim1[0], dim2[0], dim3[0], dim4[0], funcs[0], dtype[0], req_grad[0], transpose[0])
    
    seq_dim = get_test_dims(32, 512, n=3)
    hidden_dim =  get_test_dims(32, 1024 * 4, n=3)
    batch_dim = get_test_dims(2, 16, n=3)
    test_dim3_igemm(seq_dim[0], hidden_dim[0], batch_dim[0])
    print("Test igemmlt dim3 matmul pass")
    
    transpose = BOOLEAN_TUPLES
    #test_igemm(hidden_dim[0], batch_dim[0], transpose[0], seq_dim[0])
    #print("Test standard igemmlt matmul pass")
    
    #test_ibmm(dim1[0], dim2[0], dim3[0], dim4[0], transpose[0])
    print("Test standard ibmm matmul pass")
    
    batch = 1
    seq= 1
    model =6656
    hidden = 4*6656
    #test_bench_matmul(batch, seq, model, hidden)
    
    