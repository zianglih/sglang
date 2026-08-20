import unittest

import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
    w8a8_block_fp8_matmul,
    w8a8_block_fp8_matmul_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")

from sglang.srt.utils import get_device, is_cuda, is_xpu

_is_cuda = is_cuda()
_is_xpu = is_xpu()

device = get_device()


class TestFP8Base(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.M = 256
        # test non-aligned
        cls.N = 1024 + 64
        cls.K = 512
        cls.group_size = 128
        cls.quant_type = torch.float8_e4m3fn
        cls.output_type = torch.bfloat16

    @staticmethod
    def _make_A(M, K, group_size, out_dtype):
        quant_A = torch.rand(
            M, K // group_size, group_size, dtype=torch.float32, device=device
        )
        # -1 ~ 1
        quant_A = quant_A * 2 - 1
        # scaling abs max to fmax
        finfo = torch.finfo(out_dtype)
        fmax = finfo.max
        scaling = fmax / quant_A.abs().amax(-1, keepdim=True)
        quant_A *= scaling
        quant_A = quant_A.to(out_dtype).to(torch.float32)

        # create scale and A
        scale = torch.rand(M, K // group_size, dtype=torch.float32, device=device)
        scale /= fmax
        A = quant_A * scale[..., None]

        A = A.reshape(M, K)
        quant_A = quant_A.reshape(M, K).to(out_dtype)
        return A, quant_A, scale

    @staticmethod
    def _make_B(K, N, group_size, out_dtype):
        def _aligned_size(a, b):
            return (a + b - 1) // b * b

        K_aligned = _aligned_size(K, group_size)
        N_aligned = _aligned_size(N, group_size)

        quant_B = torch.rand(
            K_aligned // group_size,
            group_size,
            N_aligned // group_size,
            group_size,
            dtype=torch.float32,
            device=device,
        )
        quant_B = quant_B * 2 - 1

        # scaling abs max to fmax
        finfo = torch.finfo(out_dtype)
        fmax = finfo.max
        scaling = fmax / quant_B.abs().amax((1, 3), keepdim=True)
        quant_B *= scaling
        quant_B = quant_B.to(out_dtype).to(torch.float32)

        scale = torch.rand(
            K_aligned // group_size,
            1,
            N_aligned // group_size,
            1,
            dtype=torch.float32,
            device=device,
        )
        scale /= fmax

        B = quant_B * scale

        B = B.reshape(K_aligned, N_aligned)[:K, :N]
        quant_B = quant_B.reshape(K_aligned, N_aligned).to(out_dtype)[:K, :N]
        scale = scale.reshape(K_aligned // group_size, N_aligned // group_size)
        return B, quant_B, scale


class TestPerTokenGroupQuantFP8(TestFP8Base):
    def test_per_token_group_quant_fp8(self):
        if _is_cuda and torch.cuda.get_device_capability()[0] < 9:
            return

        A, A_quant_gt, scale_gt = self._make_A(
            M=self.M, K=self.K, group_size=self.group_size, out_dtype=self.quant_type
        )
        A_quant, scale = per_token_group_quant_fp8(
            x=A.to(torch.bfloat16), group_size=self.group_size
        )
        torch.testing.assert_close(scale, scale_gt)
        diff = (A_quant.to(torch.float16) - A_quant_gt.to(torch.float16)).abs()
        diff_count = (diff > 1e-5).count_nonzero()
        assert diff_count / diff.numel() < 1e-4


class TestW8A8BlockFP8Matmul(TestFP8Base):
    def test_w8a8_block_fp8_matmul(self):
        if _is_cuda and torch.cuda.get_device_capability()[0] < 9:
            return
        elif _is_xpu:
            # XPU doesn't provide traditional capability info like CUDA
            pass
        else:
            return

        A, A_quant_gt, A_scale_gt = self._make_A(
            M=self.M, K=self.K, group_size=self.group_size, out_dtype=self.quant_type
        )
        B, B_quant_gt, B_scale_gt = self._make_B(
            K=self.K, N=self.N, group_size=self.group_size, out_dtype=self.quant_type
        )
        C_gt = A.to(self.output_type) @ B.to(self.output_type)
        C = w8a8_block_fp8_matmul(
            A=A_quant_gt,
            B=B_quant_gt.T.contiguous(),
            As=A_scale_gt,
            Bs=B_scale_gt.T.contiguous(),
            block_size=[128, 128],
            output_dtype=self.output_type,
        )
        torch.testing.assert_close(C, C_gt, atol=0.5, rtol=1e-4)

    def test_w8a8_block_fp8_matmul_post_delta(self):
        if not _is_cuda or torch.cuda.get_device_capability()[0] < 9:
            return

        A, A_quant, A_scale = self._make_A(
            M=self.M, K=self.K, group_size=self.group_size, out_dtype=self.quant_type
        )
        _, B_quant, B_scale = self._make_B(
            K=self.K, N=self.N, group_size=self.group_size, out_dtype=self.quant_type
        )
        del A

        common = dict(
            A=A_quant,
            B=B_quant.T.contiguous(),
            As=A_scale,
            Bs=B_scale.T.contiguous(),
            block_size=[128, 128],
        )
        native = w8a8_block_fp8_matmul_triton(
            **common, output_dtype=self.output_type
        )
        slots = torch.arange(self.M, device=device, dtype=torch.int32) % 3
        zero_delta = torch.zeros(
            (3, self.N), device=device, dtype=torch.float32
        )
        identity = w8a8_block_fp8_matmul_triton(
            **common,
            output_dtype=self.output_type,
            post_delta_bank=zero_delta,
            candidate_slots=slots,
        )
        torch.testing.assert_close(identity, native, rtol=0, atol=0)

        torch.manual_seed(19)
        bias = torch.randn(self.N, device=device, dtype=torch.bfloat16) / 10
        delta = torch.randn((3, self.N), device=device, dtype=torch.float32) / 100
        accumulator = w8a8_block_fp8_matmul_triton(
            **common, output_dtype=torch.float32, bias=bias
        )
        actual = w8a8_block_fp8_matmul_triton(
            **common,
            output_dtype=self.output_type,
            bias=bias,
            post_delta_bank=delta,
            candidate_slots=slots,
        )
        expected = torch.addcmul(accumulator, accumulator, delta[slots]).to(
            self.output_type
        )
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-2)

        for graph_m in (1, 17, 128):
            graph_a = A_quant[:graph_m].clone()
            graph_as = A_scale[:graph_m].clone()
            graph_slots = slots[:graph_m].clone()
            graph_delta = delta.clone()
            graph_common = {
                **common,
                "A": graph_a,
                "As": graph_as,
            }

            # Production warms every graph bucket before capture.
            w8a8_block_fp8_matmul_triton(
                **graph_common,
                output_dtype=self.output_type,
                bias=bias,
                post_delta_bank=graph_delta,
                candidate_slots=graph_slots,
            )
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = w8a8_block_fp8_matmul_triton(
                    **graph_common,
                    output_dtype=self.output_type,
                    bias=bias,
                    post_delta_bank=graph_delta,
                    candidate_slots=graph_slots,
                )

            graph_a.copy_((-graph_a.float()).to(graph_a.dtype))
            graph_slots.copy_((graph_slots + 1) % 3)
            graph_delta.mul_(0.5)
            graph.replay()
            torch.cuda.synchronize()

            graph_accumulator = w8a8_block_fp8_matmul_triton(
                **graph_common, output_dtype=torch.float32, bias=bias
            )
            expected = torch.addcmul(
                graph_accumulator,
                graph_accumulator,
                graph_delta[graph_slots],
            ).to(self.output_type)
            torch.testing.assert_close(captured, expected, rtol=2e-3, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
