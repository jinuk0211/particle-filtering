
from easydict import EasyDict as edict
import math

cfg = edict()

cfg.note = "debug"

cfg.api = "huggingface"
cfg.allowed_apis = ["together", "huggingface", "llama", "vllm", "debug", "gpt-4o"]

cfg.seed = 42
cfg.verbose = False
cfg.tensor_parallel_size = 1
cfg.half_precision = False
# WandB settings
cfg.wandb_mode = "disabled"  # options: ["disabled", "online"]
# LLM settings
# cfg.model_ckpt = "google/gemma-3-1b-it"  # <-- 반드시 수동으로 설정해야 함
cfg.model_ckpt = "Qwen/Qwen2.5-7B-Instruct"
#cfg.model_ckpt = "meta-llama/Llama-3.2-1B-Instruct"
cfg.model_parallel = False
cfg.half_precision = False
cfg.max_tokens = 1024
cfg.temperature = 0.4
cfg.top_k = 40
cfg.top_p = 0.9
cfg.num_beams = 3
# cfg.repetition_penalty = 1.1
cfg.max_num_worker = 3
cfg.test_batch_size = 1
cfg.tensor_parallel_size = 1

# prompt settings
cfg.prompts_root = "prompts"

# dataset settings
cfg.data_root = "data"
cfg.allowed_dataset_names = [
    "FMT", "GPQA", "WICE", "CWEBQA", "MATH", "GSM8K", "GSM8KHARD",
    "STG", "SVAMP", "MULTIARITH", "ScienceQA", "SciKEval", "CFA"
]
cfg.dataset_name = "GPQA"  # <-- 반드시 실제 사용 전에 바꿔야 함
cfg.test_json_filename = "test_all"
cfg.start_idx = 0
cfg.end_idx = math.inf
# outputs settings
cfg.run_outputs_root = "run_outputs"
cfg.eval_outputs_root = "eval_outputs"

cfg.temperature = 0.8
cfg.top_p = 0.95
cfg.top_k = 40
cfg.repetition_penalty = 1.1
cfg.n = 1
cfg.max_tokens = 256
cfg.logprobs = 1
cfg.stop = []
cfg.disable_rag = True
cfg.num_subquestions = 3
cfg.num_votes = 3
cfg.max_tokens = 256
cfg.enable_potential_score = True

cfg.mcts_num_last_votes = 3


from easydict import EasyDict as edict
import math

# cfg = edict()

# cfg.note = "debug"

# cfg.api = "vllm"
# cfg.allowed_apis = ["together", "huggingface", "llama", "vllm", "debug", "gpt-4o"]

# cfg.seed = 42
# cfg.verbose = False
# cfg.tensor_parallel_size = 1
# cfg.half_precision = False
# # WandB settings
# cfg.wandb_mode = "disabled"  # options: ["disabled", "online"]
# # LLM settings
# # cfg.model_ckpt = "google/gemma-3-1b-it"  # <-- 반드시 수동으로 설정해야 함
# cfg.model_ckpt = "meta-llama/Llama-3.2-1B-Instruct"
# cfg.model_parallel = False
# cfg.half_precision = False
# cfg.max_tokens = 1024
# cfg.temperature = 0.4
# cfg.top_k = 40
# cfg.top_p = 0.9
# cfg.num_beams = 3
# # cfg.repetition_penalty = 1.1
# cfg.max_num_worker = 3
# cfg.test_batch_size = 1
# cfg.tensor_parallel_size = 1

# # prompt settings
# cfg.prompts_root = "prompts"

# # dataset settings
# cfg.data_root = "data"
# cfg.allowed_dataset_names = [
#     "FMT", "GPQA", "WICE", "CWEBQA", "MATH", "GSM8K", "GSM8KHARD",
#     "STG", "SVAMP", "MULTIARITH", "ScienceQA", "SciKEval", "CFA"
# ]
# cfg.dataset_name = "GPQA"  # <-- 반드시 실제 사용 전에 바꿔야 함
# cfg.test_json_filename = "test_all"
# cfg.start_idx = 0
# cfg.end_idx = math.inf
# # outputs settings
# cfg.run_outputs_root = "run_outputs"
# cfg.eval_outputs_root = "eval_outputs"

# cfg.temperature = 0.8
# cfg.top_p = 0.95
# cfg.top_k = 40
# cfg.repetition_penalty = 1.1
# cfg.n = 1
# cfg.max_tokens = 256
# cfg.logprobs = 1
# cfg.stop = []
# cfg.disable_rag = True
# cfg.num_subquestions = 3
# cfg.num_votes = 3
# cfg.max_tokens = 256
# cfg.enable_potential_score = True

# cfg.mcts_num_last_votes = 3
# # generator arg
#         # if not args.disable_rag:
#         #     self.retriever = Retriever()
#         #     self.retriever.regist_io_system(self.io)

#         # self.num_subquestions = args.num_subquestions
#         # self.num_a1_steps = args.num_a1_steps
#         # self.num_votes = args.num_votes
#         # self.max_tokens = args.max_tokens
#         # self.enable_potential_score = args.enable_potential_score

#         # self.mcts_num_last_votes = args.mcts_num_last_votes

#         # with open(args.decompose_template_path, "r") as f:        