from types import SimpleNamespace

import pytest
from sglang.srt.model_loader.weight_utils import maybe_add_mtp_safetensors
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _config():
    return SimpleNamespace(
        architectures=["JoyAILLMFlashForCausalLMNextN"],
        model_type="joyai_llm_flash",
        num_nextn_predict_layers=1,
    )


def test_joyai_draft_loader_replaces_target_shards_with_complete_mtp_shard(tmp_path):
    target = tmp_path / "model-00001-of-00001.safetensors"
    mtp = tmp_path / "mtp-1-of-1.safetensors"
    target.touch()
    mtp.touch()

    files = maybe_add_mtp_safetensors(
        [str(target)], str(tmp_path), "model.safetensors.index.json", _config()
    )

    assert files == [str(mtp)]


def test_joyai_draft_loader_fails_closed_without_separate_mtp_shard(tmp_path):
    target = tmp_path / "model-00001-of-00001.safetensors"
    target.touch()

    with pytest.raises(RuntimeError, match="mtp-1-of-1.safetensors"):
        maybe_add_mtp_safetensors(
            [str(target)], str(tmp_path), "model.safetensors.index.json", _config()
        )
