import vllm
import vllm.v1.worker.gpu_model_runner

from vllm_ascend.worker.v2.attn_utils import (
    _allocate_kv_cache,
    _reshape_kv_cache_v2,
    get_kv_cache_spec,
)


def _get_kv_cache_spec_for_runner(self):
    return get_kv_cache_spec(self.vllm_config)


vllm.v1.worker.gpu.attn_utils._allocate_kv_cache = _allocate_kv_cache
vllm.v1.worker.gpu.attn_utils._reshape_kv_cache = _reshape_kv_cache_v2
vllm.v1.worker.gpu.model_runner.get_kv_cache_spec = get_kv_cache_spec
vllm.v1.worker.gpu.model_runner.GPUModelRunner.get_kv_cache_spec = (
    _get_kv_cache_spec_for_runner
)
vllm.v1.worker.gpu_model_runner.GPUModelRunner.get_kv_cache_spec = (
    _get_kv_cache_spec_for_runner
)
