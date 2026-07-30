import os

from vllm import LLM, SamplingParams
from vllm.sampling_params import RepetitionDetectionParams

MAIN_MODELS = "/mnt/weight/Qwen3-8B"
# EGALE_MODELS = "vllm-ascend/EAGLE-LLaMA3.1-Instruct-8B"

def test_egale_spec_decoding(
    model: str,
    # eagle_model: str,
    max_tokens: int,
    enforce_eager: bool,
) -> None:
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        
        # "If all humans are mortal and Socrates is human, then",
        # "What is the solution to the equation 2x + 5 = 15?",
        # "Explain the concept of quantum entanglement in simple terms:",
        
        # "Write a short poem about the changing seasons:",
        # "Begin a science fiction story about humanity's first contact with aliens:",
        # "Describe a peaceful morning in a Japanese temple:",
    ] * 16
    sampling_params = SamplingParams(
        max_tokens=max_tokens,

        temperature=0.8,
        # top_p=0.9,
        # top_k=10,
        min_p=0.1,

        presence_penalty=0.2,
        frequency_penalty=0.2,
        repetition_penalty=1.1,

        seed=42,

        logit_bias={0: -1.0, 1: 0.5},
        _bad_words_token_ids=[[0], [1, 2]],
        logprobs=5,
        prompt_logprobs=1,
        flat_logprobs=True,

        repetition_detection=RepetitionDetectionParams(
            max_pattern_size=5,
            min_pattern_size=2,
            min_count=3,
        ),
    )

    llm = LLM(
        model,
        max_model_len=1024,
        enforce_eager=enforce_eager,
        async_scheduling=True,
        gpu_memory_utilization=0.8,
        
        max_num_seqs=64,
        # speculative_config={
        #     "model": eagle_model,
        #     "method": "eagle",
        #     "num_speculative_tokens": 3,
        # },

        profiler_config = {
            "profiler": "torch",
            "torch_profiler_dir": "./vllm_profile",
            "torch_profiler_with_stack": False},
        seed = 42,
    )
    
    llm.start_profile()
    llm.generate(prompts, sampling_params)
    llm.stop_profile()

def main():
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "10"
    os.environ["VLLM_VERSION"] = "0.25.1"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["VLLM_USE_MODELSCOPE"] = "True"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    test_egale_spec_decoding(MAIN_MODELS, 128, True)

if __name__ == "__main__":
    main()
