# particle-filtering
https://arxiv.org/html/2502.01618v3#S4
https://www.stats.ox.ac.uk/~doucet/andrieu_doucet_holenstein_PMCMC.pdf
https://openreview.net/pdf?id=xoXn62FzD0
```txt
pip install transformers vllm huggingface_hub
pip install fuzzywuzzy easydict datasets accelerate
huggingface-cli login
```
```bash
NUM_PARTICLES=(32 16 8 4 2 1 64 128)
HF_TOKEN={Your HF Token}


for P in ${NUM_PARTICLES[@]}; do
HF_TOKEN={HF_TOKEN}  python /new_data/probabilistic_inference_scaling/probabilistic_inference_scaling/scripts/pg.py \
        --total-timesteps 1 \
        --n-particles $P \
        --dataset-start 0 \
        --dataset-end 500 \
        --prm-path Qwen/Qwen2.5-Math-PRM-7B \
        --softmax-temp 1 \
        --seed 96 \
        --model-path meta-llama/Llama-3.1-8B-Instruct \
        --output-dir /new_data/probabilistic_inference_scaling/probabilistic_inference_scaling/llama8b_qwenRM_results_jan20/seed96/softmax_temp1/model_tempPoint8/p$P/ \
        --resample-inactive
doneNUM_PARTICLES=(32 16 8 4 2 1 64 128)
HF_TOKEN={Your HF Token}


for P in ${NUM_PARTICLES[@]}; do
HF_TOKEN={HF_TOKEN}  python /new_data/probabilistic_inference_scaling/probabilistic_inference_scaling/scripts/pg.py \
        --total-timesteps 1 \
        --n-particles $P \
        --dataset-start 0 \
        --dataset-end 500 \
        --prm-path Qwen/Qwen2.5-Math-PRM-7B \
        --softmax-temp 1 \
        --seed 96 \
        --model-path meta-llama/Llama-3.1-8B-Instruct \
        --output-dir /new_data/probabilistic_inference_scaling/probabilistic_inference_scaling/llama8b_qwenRM_results_jan20/seed96/softmax_temp1/model_tempPoint8/p$P/ \
        --resample-inactive
done
```
```python
from collections import defaultdict
from evaluator import GPQAEvaluator
from generator import Generator, load_vLLM_model, generate_with_vLLM_model, LLM
from prompt import rag_prompt, eval_prompt
import numpy as np
from cfg import cfg
from huggingface_hub import login
from datasets import load_dataset, Dataset
from verify import probability_subanswer_question, probability_subquestion_question
import torch
import time
import random
from evaluate import run_evaluation
import pandas as pd
import os, sys
sys.path.insert(0, "../pg")
from pg import particle_gibbs_kernel, particle_gibbs
from rm import load_prm
from config import Config


model_name = "Qwen/Qwen2.5-7B-Instruct"
global global_value_model, global_tokenizer
model,tokenizer=LLM(model='qwen2.5')
global_value_model = model       
global_tokenizer = tokenizer
evaluator = GPQAEvaluator()
print('llm done')

config = Config()
config.output_dir = None
config.model_path = "meta-llama/Llama-3.2-1B-Instruct"
config.prm_path = "Qwen/Qwen2.5-Math-PRM-7B"
prm = load_prm(config)
print('prm done')
generator = Generator(cfg, tokenizer, model, evaluator) ##tokenizer, model = load_vLLM_model(cfg.model_ckpt, cfg.seed, cfg.tensor_parallel_size, cfg.half_precision)

split = 100


#reranker = True
reranker = False
rag_only_one = False
critic = False
output_list = []
input_list = []
subquestions_list = []
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Subquestion generator script.")
    
    parser.add_argument(
        '--num_subquestion',
        type=int,
        default=2,
        help='Number of subquestions to generate per main question (default: 3)'
    )
    parser.add_argument(
        '--num_subanswer',
        type=int,
        default=2,
        help='Number of subanswer to generate per main question (default: 3)'
    )    
    parser.add_argument(
        '--ds',
        type=str,
        default='math',
        help='name of data'
    )        
    return parser.parse_args()
from generator import concat_subqs_and_subas

if __name__ == "__main__":
  args = parse_args()
  if args.ds == 'gpqa':
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
    df = ds['train'].to_pandas()
  elif args.ds == 'popqa':
    df = pd.read_csv("hf://datasets/akariasai/PopQA/test.tsv", sep="\t")
    df = df[:200]
    print('indexed 200')
  elif args.ds == 'arc':
    splits = {'train': 'ARC-Challenge/train-00000-of-00001.parquet', 'test': 'ARC-Challenge/test-00000-of-00001.parquet', 'validation': 'ARC-Challenge/validation-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/allenai/ai2_arc/" + splits["train"])
  elif args.ds == 'math':
    df = pd.read_json("hf://datasets/HuggingFaceH4/MATH-500/test.jsonl", lines=True)

    
  with torch.no_grad():
    global_value_model = model 
    global_tokenizer = tokenizer
    
    for row in range(1):
      if args.ds == 'gpqa':
        question = ds['train']['Question'][row]
        prompt = ds['train'][row]['Question'] + 'A)' + ds['train'][row]['Correct Answer'].replace('\n','') + ' B)' + ds['train'][row]['Incorrect Answer 1'].replace('\n','') + ' C)' + ds['train'][row]['Incorrect Answer 2'].replace('\n','') + ' D)' + ds['train'][row]['Incorrect Answer 3'].replace('\n','')
      elif args.ds == 'popqa':
        question = df['question'][row]
        prompt = question 
      elif args.ds == 'arc':
        question = df['question'][row]
        prompt = question + "\nA)" + df['choices'][row]['text'][0] + " B)" + df['choices'][row]['text'][1] + " C)" + df['choices'][row]['text'][2] + " D)" + df['choices'][row]['text'][3]
      elif args.ds == 'math':
        question = df['problem'][row]
      

      input_list.append(question)
      value_list = []
      final_questions = []
      subquestions_retrieval=[]
      best_subquestion = None  
      subq_list = []
      suba_list = []
      
      particle_intermediate_storage = []
      particles_tracker = []
      current_timestep = 1
            
      def gibbs_kernel(question):
      stepwise_particle_tracker_before = []
      stepwise_particle_tracker_after = []
  
      # Initialize particles
      if reference_particle is None:
          particles = [Particle(temperature=llm_sampling_temp) for _ in range(n_particles)]
      else:
          particles = [Particle(temperature=llm_sampling_temp) for _ in range(n_particles - 1)]
  
      print(f"Initialized {n_particles} particles.")
      
      #responses, stops = take_a_step_for_batch(question, llm, config, first=True, temperature=llm_sampling_temp, n_particles=len(particles))
      subquestions, subqids = generator.subq(question,subq_list,suba_list,num_subq=5)
      for i in subquestions:
        value = probability_subquestion_question(question, subquestions) # probablistic score
        value_list.append(value)
      print(f'values of subquestions{value_list}')
      top_indices = np.argsort(value_list)[::-1][:1]
      top_subquestion = [subquestions[i] for i in top_indices] #top3 subquestions
      subq_list.append(subquestions[0].split('subanswer')[0])
      print(f'subq after subq:{subq_list}')
      subanswers, subaids = generator.suba(question,subq_list,suba_list,num_suba = 1)
      
      print(f'ids:{subaids}')
      suba_list.append(subanswers[0].split('subquestion')[0])      
      #-------------------- first done
      subquestions, subqids = generator.subq(question,subq_list,suba_list,num_subq=5)
      for i in subquestions:
        value = probability_subquestion_question(question, subquestions) # probablistic score
        value_list.append(value)
      print(f'values of subquestions{value_list}')
      top_indices = np.argsort(value_list)[::-1][:1]
      top_subquestion = [subquestions[i] for i in top_indices] #top3 subquestions
      subq_list.append(subquestions[0].split('subanswer')[0])  
      
      for particle in particles:    
        particle.subq_list.append(top_subquestion[0].split('subanswer')[0])
          
      for idx, particle in enumerate(particles):
          #response, stop = take_a_step(question, llm, config, first=True, temperature=llm_sampling_temp)
          subanswers, subaids = generator.suba(question,particle.subq_list,particle.suba_list,num_suba = 1)
          stop = stop_reason(subaids[0].tolist())
          #reward = prm.score([question], [[]])[-1][-1][-1]
          reward = prm.score([subanswers[0]], [[]])[-1][-1][-1]
          #particle.add_step(response, reward, stop)
          particle.add_step([subanswers[0]],reward, stop)
          particle.suba_list.append([subanswers[0]])
          
      
  
      stepwise_particle_tracker_before.append([p.deepcopy() for p in particles])
  
      step = 1
  
      while any(particle.is_active() for particle in particles):
          
          rewards = [particle.get_last_reward() for particle in particles]

          if reference_particle is not None:
              if step >= len(reference_particle.rewards):
                  rewards.append(reference_particle.rewards[-1])
              else:
                  rewards.append(reference_particle.rewards[step])

          logits = [inverse_sigmoid(r) for r in rewards]
          logits = np.array(logits)
          
          if temperature_annealing:
              softmax_temp = temperature_linear_annealing(
                  starting_temp=temperature_annealing[0],
                  ending_temp=temperature_annealing[1],
                  total_steps=temperature_annealing[2],
                  current_step=step,
              )

          logger.info(f"Softmax temperature: {softmax_temp}")
          weights = softmax(logits / softmax_temp)


          # Sample new particles based on weights
          if reference_particle is None:
              sampled_particles = np.random.choice(
                  particles,
                  size=len(particles),
                  p=weights,
                  replace=True,  # before particles was active_particles
              )
          else:
              sampled_particles = np.random.choice(
                  particles + [reference_particle],
                  size=len(particles),
                  p=weights,
                  replace=True,  # before particles was active_particles
              )

          particles = [
              particle.deepcopy(numSteps=step) for particle in sampled_particles
          ]
  
          stepwise_particle_tracker_after.append([p.deepcopy() for p in particles])
          subq_tmp = subq_list
          suba_tmp = suba_list
          # Take a step for each sampled particle
          for idx, particle in enumerate(particles):
          
              subquestions, subqids = generator.subq(question,particle.subq_list,particle.suba_list,num_subq=5)
              for i in subquestions:
                value = probability_subquestion_question(question, subquestions) # probablistic score
                value_list.append(value)
              print(f'values of subquestions{value_list}')
              top_indices = np.argsort(value_list)[::-1][:1]
              top_subquestion = [subquestions[i] for i in top_indices] #top3 subquestions
              particle.subq_list.append(top_subquestion[0].split('subanswer')[0])
                 
              if not particle.is_active():
                  continue
                  
              #response, stop = take_a_step(question, llm, config, first=False, steps_so_far=particle.trajectory)
              subanswers, subaids = generator.suba(question,particle.subq_list,particle.suba_list,num_suba = 1)
              stop = stop_reason(subaids[0].tolist())
              response_to_pass_for_score = "\n\n".join(particle.trajectory) + "\n\n" + subanswers[0]
              
              reward = prm.score([question], [[response_to_pass_for_score]])[-1][-1][-1]
             
              particle.add_step(response, reward, stop)
             
          subq_list.append(subquestions[0].split('subanswer')[0])
          step = step + 1
          stepwise_particle_tracker_before.append([p.deepcopy() for p in particles])
  
      if reference_particle is None:
          return particles, stepwise_particle_tracker_before, stepwise_particle_tracker_after 
      else:
          return particles + [reference_particle], stepwise_particle_tracker_before, stepwise_particle_tracker_after
      #def gibbs kernel end
      particles_tracker.append(current_particles)
      particle_intermediate_storage.append([copy.deepcopy(current_particles)])
  
      # compute_budget_used = sum([len(p.trajectory) for p in current_particles])
      if total_timesteps > 1:
          while current_timestep < total_timesteps:
              # preferred_particle = current_particles[
              #     np.argmax([p.rewards[-1] for p in current_particles])
              # ]
              rewards = [particle.get_last_reward() for particle in current_particles]
  
              logits = [inverse_sigmoid(r) for r in rewards]
              logits = np.array(logits)
  
              if temperature_annealing:
                  softmax_temp = temperature_linear_annealing(
                      starting_temp=temperature_annealing[0],
                      ending_temp=temperature_annealing[1],
                      total_steps=temperature_annealing[2],
                      current_step=current_timestep,
                  )
  
              logger.info(f"Softmax temperature: {softmax_temp}")
              weights = softmax(logits / softmax_temp)
  
              preferred_particle = np.random.choice(
                  current_particles,
                  size=1,
                  p=weights,
                  replace=True,                             # before particles was active_particles
              )[0]
  
              preferred_particle.preferred = True      
              current_particles, tracker_before, tracker_after = particle_gibbs_kernel(
                  x["problem"],
                  llm,
                  prm,
                  config,
                  n_particles,
                  softmax_temp,
                  resample_inactive=resample_inactive,
                  reference_particle=preferred_particle,
                  temperature_annealing=temperature_annealing
              )
              particles_tracker.append(current_particles)
              current_timestep += 1
              particle_intermediate_storage.append([copy.deepcopy(current_particles)])
  
      # # Create output directory if it doesn't exist
      os.makedirs(config.output_dir, exist_ok=True)
  
      save_path = os.path.join(config.output_dir, f"{question_id}.pkl")
      with open(save_path, "wb") as f:
          pickle.dump(particles_tracker, f)
  
      intermediate_dir = os.path.join(config.output_dir, "intermediate")
      os.makedirs(intermediate_dir, exist_ok=True)
      save_path_intermediate = os.path.join(intermediate_dir, f"{question_id}_intermediate.pkl")
      with open(save_path_intermediate, "wb") as f:
          pickle.dump(particle_intermediate_storage, f)
          
      #with open(save_path.replace(".pkl", "_before.pkl"), "wb") as f:
      #    pickle.dump(tracker_before, f)
      
      #with open(save_path.replace(".pkl", "_after.pkl"), "wb") as f:
      #    pickle.dump(tracker_after, f)
  
      logger.info(f"Saved particles to: {save_path}")
  
      return x
      
            
  


        if 'Now we can' in subanswers[0]:
          print('done')
          break
      output = generator.final_output(question,subq_list,suba_list)
      output_list.append(output)
    #result = pd.DataFrame(output)
    #result.to_csv(f'output_{args.ds}.csv', index=False, encoding="utf-8")
```
```python
@dataclass
class Config:
    approach: Literal["best_of_n", "beam_search", "dvts"] = "best_of_n"
    model_path: str = "meta-llama/Llama-3.2-1B-Instruct"
    gpu_memory_utilization: float = (
        0.5  # vllm is allocated 0.5 of GPU memory, the PRM uses the rest
    )
    prm_path: str = "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"
    # Output Related Options
    output_dir: str = None
    num_proc: int = None
    push_to_hub: bool = False
    hub_dataset_id: str = None
    overwrite_hub_revision: bool = False
    apply_voting: bool = True
    use_continuous_batching: bool = True

    # Dataset Related Options
    dataset_name: str = "datasets/math500.jsonl"
    dataset_config: str = None
    dataset_split: str = "test"
    dataset_start: int = None
    dataset_end: int = None
    num_samples: int = None

    # Chat template related options
    system_prompt: str = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem."
    custom_chat_template: str = '{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now("%d %b %Y") %}\n    {%- else %}\n        {%- set date_string = "26 Jul 2024" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{{- "Cutting Knowledge Date: December 2023\\n" }}\n{{- "Today Date: " + date_string + "\\n\\n" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n        {{- \'{"name": "\' + tool_call.name + \'", \' }}\n        {{- \'"parameters": \' }}\n        {{- tool_call.arguments | tojson }}\n        {{- "}" }}\n        {{- "<|eot_id|>" }}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n'
    # Search Related Options
    n: int = 4
    temperature: float = 0.8
    top_p: float = 1.0
    prm_batch_size: int = 4
    search_batch_size: int = 25
    seed: int = 42
    max_tokens: int = 2048
    agg_strategy: str = "last"  # Options: "last", "min", "prod"

    # DVTS / Beam Search options
    beam_width: int = 4  # m in the paper
    num_iterations: int = 40
    lookahead: int = 1

    # Beam search options:
    filter_duplicates: bool = False
    sort_completed: bool = False
```
pg.py
------

```python

import logging
import time
import torch
import click
from vllm import LLM

from datasets import load_dataset
from sal.config import Config
from sal.models.reward_models import load_prm
from sal.search import particle_gibbs
from sal.search.particle_gibbs_batch import particle_gibbs_batch
from sal.utils.data import get_dataset, save_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import os
# Check CUDA_DEVICE_MAX_CONNECTIONS environment variable
logger.info(f"CUDA_DEVICE_MAX_CONNECTIONS: {os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS')}")

@click.command()
@click.option(
    "--dataset-start",
    default=0,
    type=int,
    help="Start index of the dataset to process.",
    show_default=True,
)
@click.option(
    "--dataset-end",
    default=38,
    type=int,
    help="End index of the dataset to process.",
    show_default=True,
)
@click.option(
    "--seed",
    default=None,
    # type=int,
    help="Random seed for reproducibility.",
    show_default=True,
)
@click.option(
    "--total-timesteps",
    default=1,
    type=int,
    help="Total timesteps for particle Gibbs sampling.",
    show_default=True,
)
@click.option(
    "--n-particles",
    default=4,
    type=int,
    help="Number of particles for Gibbs sampling.",
    show_default=True,
)
@click.option(
    "--softmax-temp",
    default=1.0,
    type=float,
    help="Softmax temperature for sampling.",
    show_default=True,
)
@click.option(
    "--llm-sampling-temp",
    default=0.8, 
    type=float,
    help="Temperature for LLM sampling.",
    show_default=True,
)
@click.option(
    "--temperature-annealing",
    default=None,
    type=(float, float, int),
    help="Parameters for temperature annealing (start_temp, end_temp, total_steps).",
)
@click.option(
    "--resample-inactive",
    is_flag=True,
    default=False,
    help="Whether to resample inactive particles.",
    show_default=True,
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(file_okay=False, writable=True),
    help="Output directory to save the results.",
)
@click.option(
    "--model-path",
    default="meta-llama/Llama-3.2-1B-Instruct",
    type=str,
    help="Path to the language model.",
    show_default=True,
)
@click.option(
    "--prm-path",
    default="RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
    type=str,
    help="Path to the probabilistic reward model.",
    show_default=True,
)
@click.option(
    "--dataset-path",
    default=None,
    type=str,
    help="Path to the dataset.",
    show_default=True,
)
def main(
    dataset_start,
    dataset_end,
    seed,
    total_timesteps,
    n_particles,
    llm_sampling_temp,
    softmax_temp,
    temperature_annealing,
    resample_inactive,
    output_dir,
    model_path,
    prm_path,
    dataset_path,
):
    """
    Run Particle Gibbs sampling for a dataset using a specified LLM and reward model.
    """
    start_time = time.time()  # Start timer

    enable_prefix_caching = False

    # Log all the arguments
    logger.info("Starting execution with the following parameters:")
    logger.info(f"Dataset start: {dataset_start}")
    logger.info(f"Dataset end: {dataset_end}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Total timesteps: {total_timesteps}")
    logger.info(f"Number of particles: {n_particles}")
    logger.info(f"LLM sampling temperature: {llm_sampling_temp}")
    logger.info(f"Softmax temperature: {softmax_temp}")
    logger.info(f"Temperature annealing: {temperature_annealing}")
    logger.info(f"Resample inactive: {resample_inactive}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Enable prefix caching: {enable_prefix_caching}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"PRM path: {prm_path}")
    logger.info(f"Dataset path: {dataset_path}")

    config = Config()
    config.output_dir = output_dir
    config.model_path = model_path
    config.prm_path = prm_path

    # Initialize LLM with available GPUs
    num_gpus = 4 if "qwen2" in config.model_path.lower() else torch.cuda.device_count()

    if seed == None or seed == "None":
        logger.info("Initializing LLM without seed")
        llm = LLM(
            model=config.model_path,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            tensor_parallel_size=num_gpus,
        )
    else:
        seed = int(seed)
        logger.info(f"Initializing LLM with seed: {seed}")
        llm = LLM(
            model=config.model_path,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            seed=seed,
            tensor_parallel_size=num_gpus,
        )

    # Load the probabilistic reward model
    prm = load_prm(config)

    # Load and preprocess the dataset
    if dataset_path is not None:
        dataset = load_dataset("json", data_files=dataset_path, split="train")
    else:
        dataset = get_dataset(config)

    dataset = dataset.select(range(dataset_start, dataset_end))
    # Filter the dataset to include only ones not already completed in the directory.
    if output_dir is not None:
        logger.info(f"Filtering dataset to exclude already completed problems in {output_dir}")
        if isinstance(dataset[0]['unique_id'], int):
            dataset = dataset.filter(lambda x: not os.path.exists(os.path.join(output_dir, f"{x['unique_id']}.pkl")))
        else:
            dataset = dataset.filter(lambda x: not os.path.exists(os.path.join(output_dir, f"{x['unique_id'].replace('/', '_')[:-5]}.pkl")))

#amc2023
#{"unique_id":3,"problem":"What is the value of\n\\[2^3 - 1^3 + 4^3 - 3^3 + 6^3 - 5^3 + \\dots + 18^3 - #17^3\\]",
#"answer":3159.0,"url":"https:\/\/artofproblemsolving.com\/wiki\/index.php\/2023_AMC_12A_Problems\/Problem_12",
#"question":"What is the value of\n\\[2^3 - 1^3 + 4^3 - 3^3 + 6^3 - 5^3 + \\dots + 18^3 - 17^3?\\]"}
#math500
#{"problem":"How many elements are in the intersection of the set of all the prime numbers less than 30 and the set of all the odd numbers greater than zero?",
#"solution":"In other words, we're looking for the number of positive odd prime numbers less than 30. We go through all odd numbers less than 30 and note how many of them are prime. We get that 3, 5, 7, 11, 13, 17, 19, 23, and 29 are all of the positive odd prime numbers less than 30, a total of $\\boxed{9}$ elements in the intersection.",
#"answer":"9","subject":"Number Theory","level":2,"unique_id":"test\/number_theory\/914.json"}


    logger.info(f"Length of dataset: {len(dataset)}")

    # Perform Particle Gibbs sampling on the dataset
    dataset = dataset.map(
        particle_gibbs_batch if config.use_continuous_batching else particle_gibbs,
        batched=False,
        batch_size=1,
        fn_kwargs={
            "config": config,
            "llm": llm,
            "prm": prm,
            "total_timesteps": total_timesteps,
            "n_particles": n_particles,
            "llm_sampling_temp": llm_sampling_temp,
            "softmax_temp": softmax_temp,
            "temperature_annealing": temperature_annealing,
            "resample_inactive": resample_inactive,
        },
        desc="Running search",
        load_from_cache_file=False,
    )

    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time
    logger.info(f"Done 🔥! Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
```

load_prm
-------

```python

def load_prm(config: Config) -> PRM:
    if config.prm_path == "peiyi9979/math-shepherd-mistral-7b-prm":
        return MathShepherd(config)

    if config.prm_path == "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data":
        return RLHFFlow(config)
    if config.prm_path == "PRIME-RL/EurusPRM-Stage2":
        return PRIME(config)

    if config.prm_path == "Qwen/Qwen2.5-Math-PRM-7B":
        return QWEN_PRM(config)

    raise NotImplementedError(f"PRM {config.prm_path} not implemented")

from itertools import accumulate

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModel
)

from sal.config import Config
import torch.nn.functional as F
import re

CANDIDATE_TOKENS = [648, 387]
STEP_TAG_ID = 12902



class PRM:
    def __init__(self, search_config: Config, **model_kwargs):
        self.search_config = search_config
        if search_config.prm_path == "PRIME-RL/EurusPRM-Stage2":
            self.model, self.ref_model, self.tokenizer = self.load_model_and_tokenizer(**model_kwargs)
        else:
            self.model, self.tokenizer = self.load_model_and_tokenizer(**model_kwargs)

    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        raise NotImplementedError

    def score(
        self, questions: list[str], outputs: list[list[str]]
    ) -> list[list[float]]:
        raise NotImplementedError


class QWEN_PRM(PRM):
    def __init__(self, search_config: Config, **model_kwargs):
        super().__init__(search_config, **model_kwargs)
        self.model, self.tokenizer = self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        model_name = "Qwen/Qwen2.5-Math-PRM-7B"
        model = AutoModel.from_pretrained(model_name,
                                        device_map="auto",
                                        torch_dtype=torch.bfloat16,
                                        trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    
    def score(
        self, questions: list[str], outputs: list[list[str]], outputs_is_single_step: bool = True
    ) -> list[list[float]]:
        '''
        Score a batch of questions and their step-by-step outputs using PRIME scoring.
        questions: list of questions
        outputs: list of lists of N responses, where N answers correspond to 1 question. 
        '''
        # define a helper function. 
        def make_step_rewards(logits, token_masks):
            probabilities = F.softmax(logits, dim=-1)
            probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
            
            all_scores_res = []
            for i in range(probabilities.size(0)):
                sample = probabilities[i] # seq_len, num_labels
                positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
                non_zero_elements_list = positive_probs.cpu().tolist()
                all_scores_res.append(non_zero_elements_list)
            return all_scores_res

        # TODO: implement QWEN-PRM scoring
        all_scores = []
        for question, answers in zip(questions, outputs, strict=True):
            all_step_scores = []
            for ans in answers:
                single_step_score = []
                # we assume here that the answers use "\n\n" to separate steps. 
                if outputs_is_single_step:
                    ans = re.sub(r'\n+', '\n', ans)

                steps_list = ans.split("\n\n")
                QWEN_PRM_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
                messages = [
                    {"role": "system", "content": QWEN_PRM_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": "<extra_0>".join(steps_list) + "<extra_0>"},
                ]

                # Prepare conversation for scoring
                conversation = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )

                input_ids = self.tokenizer.encode(
                    conversation, 
                    return_tensors="pt", 
                ).to(self.model.device)

                outputs = self.model(input_ids=input_ids)

                # get the step scores
                step_sep_id = self.tokenizer.encode("<extra_0>")[0]
                token_masks = (input_ids == step_sep_id)
                step_scores = make_step_rewards(outputs[0], token_masks)

                # make the scores cumulative through multiplication
                # step_scores = [math.prod(step_scores[:i+1]) for i in range(len(step_scores))]

                all_step_scores.extend(step_scores)

            all_scores.append(all_step_scores)

        return all_scores



class PRIME(PRM):
    def __init__(self, search_config: Config, **model_kwargs):
        # override original init, because we need to load two models and a tokenizer
        super().__init__(search_config, **model_kwargs)
        self.model, self.ref_model, self.tokenizer = self.load_model_and_tokenizer()


    def load_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizer]:
        # Load PRM model
        model = AutoModelForCausalLM.from_pretrained(
            'PRIME-RL/EurusPRM-Stage2',
            device_map="auto",
            attn_implementation="flash_attention_2", 
            torch_dtype=torch.float16,
        ).eval()

        # Load reference model
        ref_model = AutoModelForCausalLM.from_pretrained(
            'Qwen/Qwen2.5-Math-7B-Instruct',
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
        ).eval()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained('PRIME-RL/EurusPRM-Stage2')

        return model, ref_model, tokenizer
    
    def score(
        self, questions: list[str], outputs: list[list[str]]
    ) -> list[list[float]]:
        '''
        Score a batch of questions and their step-by-step outputs using PRIME scoring.
        questions: list of questions
        outputs: list of lists of N responses, where N answers correspond to 1 question. 
        '''



        # TODO: implement PRIME scoring
        # implement based on the commented example code above, and also the MathShepherd code
        # Prepare inputs by combining questions and outputs
        all_scores = []
        for question, answers in zip(questions, outputs, strict=True):
            all_step_scores = []
            for ans in answers:
                single_step_score = []
                # we assume here that the answers use "\n\n" to separate steps. 
                ans_list = ans.split("\n\n")
                # Prepare conversation for scoring
                conversation = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": "\n\n".join(ans_list)},
                ]

                # Tokenize full conversation
                input_ids = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors='pt'
                ).to(self.model.device)
                
                attention_mask = (input_ids != self.tokenizer.pad_token_id).to(self.model.device)

                # Get token positions for each step
                step_last_tokens = []
                for step_num in range(0, len(ans_list) + 1):
                    step_conv = [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": "\n\n".join(ans_list[:step_num])},
                    ]
                    conv_text = self.tokenizer.apply_chat_template(
                        step_conv,
                        tokenize=False,
                        add_generation_prompt=False
                    ).strip()
                    
                    if step_num != 0 and step_num != len(ans_list):
                        conv_text += '\n\n'
                        
                    curr_ids = self.tokenizer.encode(conv_text, add_special_tokens=False)
                    step_last_tokens.append(len(curr_ids) - 2)

                inputs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': input_ids
                }

                label_mask = torch.zeros_like(input_ids)
                label_mask[0, step_last_tokens[0]:] = 1
                step_last_tokens = torch.tensor([step_last_tokens]).to(self.model.device)

                def get_logps(model,inputs):
                    logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).logits
                    labels = inputs['labels'][:, 1:].clone().long()
                    logits = logits[:, :-1, :]
                    labels[labels == -100] = 0
                    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
                    return per_token_logps

                # Get log probabilities from both models
                with torch.no_grad():
                    # Main model logprobs
                    per_token_logps = get_logps(self.model, inputs)
                    ref_per_token_logps = get_logps(self.ref_model, inputs)


                # Calculate rewards
                raw_reward = per_token_logps - ref_per_token_logps
                beta_reward = 0.001 * raw_reward * label_mask[:, 1:]  # Using 0.001 as default coefficient
                beta_reward = beta_reward.cumsum(-1)
                step_rewards = beta_reward.gather(dim=-1, index=step_last_tokens[:, 1:]).tolist()[0]
                
                all_step_scores.append(step_rewards)
            
            all_scores.append(all_step_scores)

        return all_scores






class MathShepherd(PRM):
    def load_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        model_id = "peiyi9979/math-shepherd-mistral-7b-prm"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # For batched inference
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
        ).eval()
        return model, tokenizer

    def score(
        self, questions: list[str], outputs: list[list[str]]
    ) -> list[list[float]]:
        inputs_for_prm = []
        lengths = []
        for question, output in zip(questions, outputs):
            prompt = self.search_config.system_prompt + "\n" + question + "\n"
            special_outputs = [o.replace("\n\n", " ки\n\n") for o in output]
            special_outputs = [
                o + " ки" if o[-2:] != "\n\n" else o for o in special_outputs
            ]
            inputs_for_prm.extend([f"{prompt} {o}" for o in special_outputs])
            lengths.append(len(output))

        # TODO: tokenize each batch independently so there is less padding and faster inference
        output_scores = batched_math_shepherd_inference(
            self.model,
            self.tokenizer,
            inputs_for_prm,
            self.search_config.prm_batch_size,
        )
        cumulative_lengths = list(accumulate(lengths))
        # reshape the output scores to match the input
        output_scores = [
            output_scores[i:j]
            for i, j in zip([0] + cumulative_lengths[:-1], cumulative_lengths)
        ]

        # stripped_output_scores = [] TODO: strip out the reward for previous steps
        for output_score, output in zip(output_scores, outputs):
            assert len(output_score) == len(
                output
            ), f"{len(output_score)} != {len(output)}"

        return output_scores
def batched_math_shepherd_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    inputs: list[str],
    batch_size: int,
) -> list[list[float]]:
    output_scores = []
    for i in range(0, len(inputs), batch_size):
        inputs_batch = inputs[i : i + batch_size]
        inputs_batch = tokenizer(inputs_batch, padding=True, return_tensors="pt").to(
            model.device
        )
        with torch.no_grad():
            logits = model(**inputs_batch).logits[:, :, CANDIDATE_TOKENS]
            scores = logits.softmax(dim=-1)[:, :, 0]
            step_scores_flat = scores[inputs_batch.input_ids == STEP_TAG_ID].tolist()
            # Split scores into sublist based on number of \n in the input
            step_scores = []
            counter = 0
            for i in range(len(inputs_batch.input_ids)):
                count = inputs_batch.input_ids[i].tolist().count(STEP_TAG_ID)
                step_scores.append(step_scores_flat[counter : counter + count])
                counter += count

        # Store the step scores for this batch
        output_scores.extend(step_scores)

        # Clear GPU memory
        del inputs_batch, logits, scores
        torch.cuda.empty_cache()

    return output_scores

class RLHFFlow(PRM):
    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(
            "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"
        )
        model = AutoModelForCausalLM.from_pretrained(
            "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        ).eval()
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        plus_tag_id = tokenizer.encode("+")[-1]
        minus_tag_id = tokenizer.encode("-")[-1]
        self.candidate_tokens = [plus_tag_id, minus_tag_id]

        return model, tokenizer

    def score(
        self,
        questions: list[str],
        outputs: list[list[str]],
        batched: bool = True,
        batch_size=8,
    ) -> list[list[float]]:
        if batched is True:
            return self._score_batched(questions, outputs, batch_size=batch_size)
        else:
            return self._score_single(questions, outputs)

    def _score_single(self, questions: list[str], outputs: list[list[str]]):
        # reference code: https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/math-rm/prm_evaluate.py
        all_scores = []
        for question, answers in zip(questions, outputs, strict=True):
            all_step_scores = []
            for ans in answers:
                single_step_score = []
                conversation = []
                ans_list = ans.split("\n\n")
                for k in range(len(ans_list)):
                    if k == 0:
                        # TODO: add the system prompt like we did for math shepard?
                        text = question + " " + ans_list[0]
                    else:
                        text = ans_list[k]
                    conversation.append({"content": text, "role": "user"})
                    conversation.append({"content": "+", "role": "assistant"})
                    input_ids = self.tokenizer.apply_chat_template(
                        conversation, return_tensors="pt"
                    ).to(self.model.device)
                    with torch.no_grad():
                        logits = self.model(input_ids).logits[
                            :, -3, self.candidate_tokens
                        ]  # simple version, the +/- is predicted by the '-3' position
                        step_scores = logits.softmax(dim=-1)[
                            :, 0
                        ]  # 0 means the prob of + (1 mean -)
                        # print(scores)
                        single_step_score.append(
                            step_scores[0]
                            .detach()
                            .to("cpu", dtype=torch.float32)
                            .item()
                        )

                all_step_scores.append(single_step_score)
            all_scores.append(all_step_scores)
        return all_scores

    def _score_batched(
        self, questions: list[str], outputs: list[list[str]], batch_size: int = 2
    ):
        # The RLHFlow models are trained to predict the "+" or "-" tokens in a dialogue, but since these are not unique
        # we need to introduce a dummy special token here for masking.

        special_tok_id = self.tokenizer("ки", return_tensors="pt").input_ids[0, 1]
        # We construct two parallel dialogues, one with a "+" token per assistant turn, the other with the dummy token "ки" for masking
        conversations = []
        conversations2 = []
        for question, answers in zip(questions, outputs, strict=True):
            for ans in answers:
                conversation = []
                conversation2 = []
                ans_list = ans.split("\n\n")
                for k in range(len(ans_list)):
                    if k == 0:
                        text = question + " " + ans_list[0]
                    else:
                        text = ans_list[k]
                    conversation.append({"content": text, "role": "user"})
                    conversation.append({"content": "+", "role": "assistant"})

                    # we track to location of the special token with ки in order to extract the scores
                    conversation2.append({"content": text, "role": "user"})
                    conversation2.append({"content": "ки", "role": "assistant"})

                conversations.append(conversation)
                conversations2.append(conversation2)

        output_scores = []
        for i in range(0, len(conversations), batch_size):
            convs_batch = conversations[i : i + batch_size]
            convs2_batch = conversations2[i : i + batch_size]
            inputs_batch = self.tokenizer.apply_chat_template(
                convs_batch, padding=True, return_tensors="pt"
            ).to(self.model.device)
            inputs2_batch = self.tokenizer.apply_chat_template(
                convs2_batch, padding=True, return_tensors="pt"
            ).to(self.model.device)
            assert inputs_batch.shape == inputs2_batch.shape
            with torch.no_grad():
                logits = self.model(inputs_batch).logits[:, :, self.candidate_tokens]
                scores = logits.softmax(dim=-1)[
                    :, :, 0
                ]  # 0 means the prob of + (1 mean -)

                for i in range(len(convs_batch)):
                    # We slice on the N-1 token since the model is trained to predict the Nth one ("+" in this case)
                    step_scores_flat = scores[i, :-1][
                        inputs2_batch[i, 1:] == special_tok_id
                    ].tolist()
                    output_scores.append(step_scores_flat)

        # reshape the output scores to match the input
        reshaped_output_scores = []
        counter = 0
        for question, answers in zip(questions, outputs):
            scores = []
            for answer in answers:
                scores.append(output_scores[counter])
                counter += 1
            reshaped_output_scores.append(scores)

        return reshaped_output_scores
```
```python

class Particle:
    def __init__(self, temperature=0.8):
        """
        Initializes a particle with a given temperature.
        
        Args:
            temperature (float): The initial temperature of the particle.
        """
        self.trajectory = []  # Tracks the sequence of responses
        self.rewards = []  # Tracks rewards for each step
        self.steps = 0  # Steps taken by the particle
        self.active = True  # Indicates if the particle is still evolving
        self.preferred = False  # Indicates if the particle is preferred
        self.temperature = temperature  # Dynamic temperature of the particle

    def add_step(self, response, reward, stop):
        """Adds a step to the particle's trajectory."""
        self.trajectory.append(response)
        self.rewards.append(reward)
        self.steps += 1
        if stop == "EOS" or """\\boxed""" in response:
            self.active = False
        if self.steps >= 40:
            self.active = False

    def get_last_reward(self):
        """Returns the last recorded reward."""
        return self.rewards[-1]

    def is_active(self):
        """Checks if the particle is active."""
        return self.active

    def get_trajectory(self):
        """Returns the full trajectory as a single string."""
        return "\n\n".join(self.trajectory)

    def set_temperature(self, new_temperature):
        """Sets a new temperature for the particle."""
        self.temperature = new_temperature

    def deepcopy(self, numSteps=None):
        """Returns a deep copy of the particle."""
        new_particle = Particle(temperature=self.temperature)

        if numSteps is not None:
            if numSteps >= len(
                self.trajectory
            ):  # capping it so it doesnt go out of bounds
                numSteps = len(self.trajectory)

        if numSteps is not None:
            new_particle.trajectory = self.trajectory[:numSteps]
            new_particle.rewards = self.rewards[:numSteps]
            new_particle.steps = numSteps
            if numSteps == len(self.trajectory):
                new_particle.active = self.active
            else:
                new_particle.active = True
        else:
            new_particle.trajectory = self.trajectory.copy()
            new_particle.rewards = self.rewards.copy()
            new_particle.steps = self.steps
            new_particle.active = self.active

        new_particle.preferred = self.preferred
        return new_particle

#--------------------------------------------------------------------------
def particle_gibbs(
    x,
    config,
    llm,
    prm,
    total_timesteps=1,
    n_particles=4,
    resample_inactive=True,
    softmax_temp=1.0,
    temperature_annealing=(),
    llm_sampling_temp=0.8,
):
    
    particle_intermediate_storage = []
    question_id = x["unique_id"].replace("/", "_").strip(".json")
    logger.info(f"Processing question: {question_id}")
    particles_tracker = []
    current_timestep = 1
    current_particles, tracker_before, tracker_after = particle_gibbs_kernel(
                            x["problem"], 
                            llm, 
                            prm, 
                            config, 
                            n_particles, 
                            resample_inactive=resample_inactive,
                            softmax_temp=softmax_temp,
                            temperature_annealing=temperature_annealing,
                            llm_sampling_temp=llm_sampling_temp,
    )
    particles_tracker.append(current_particles)
    particle_intermediate_storage.append([copy.deepcopy(current_particles)])

    # compute_budget_used = sum([len(p.trajectory) for p in current_particles])
    if total_timesteps > 1:
        while current_timestep < total_timesteps:
            # preferred_particle = current_particles[
            #     np.argmax([p.rewards[-1] for p in current_particles])
            # ]
            rewards = [particle.get_last_reward() for particle in current_particles]

            logits = [inverse_sigmoid(r) for r in rewards]
            logits = np.array(logits)
#--------------------------------------------
def inverse_sigmoid(x):
    """
    Calculate the inverse sigmoid (logit) of a value x.
    Args: x (float): Input value between 0 and 1 (exclusive)
    Returns: float: The inverse sigmoid value
    Raises: ValueError: If x is not between 0 and 1
    """
    # Add small epsilon to prevent log(0)
    eps = np.finfo(float).eps
    x = np.clip(x, eps, 1 - eps)
#-------------------------------------------------
    return np.log(x) - np.log(1 - x)  # More stable than np.log(x/(1-x))


            if temperature_annealing:
                softmax_temp = temperature_linear_annealing(
                    starting_temp=temperature_annealing[0],
                    ending_temp=temperature_annealing[1],
                    total_steps=temperature_annealing[2],
                    current_step=current_timestep,
                )
#------------------------------------------------------
def temperature_linear_annealing(starting_temp, ending_temp, total_steps, current_step):
    """
    Computes the temperature at a given step using linear annealing.
    Args:
        starting_temp (float): Initial temperature.
        ending_temp (float): Final temperature.
        total_steps (int): Total number of annealing steps.
        current_step (int): Current step number (0-indexed).
    Returns:
        float: Temperature at the current step.
    """
    if current_step < 0:
        raise ValueError("current_step must be >= 0.")
    if current_step >= total_steps:
        # Return constant ending temperature after the total steps.
        return ending_temp
    if total_steps <= 1:
        # Return ending temperature directly if there's only 1 or no step.
        return ending_temp

    temp_range = starting_temp - ending_temp
    step_fraction = current_step / (total_steps - 1)  # Adjust for 0-indexing
    return starting_temp - (temp_range * step_fraction)
#------------------------------------------------------

            logger.info(f"Softmax temperature: {softmax_temp}")
            weights = softmax(logits / softmax_temp)

            preferred_particle = np.random.choice(
                current_particles,
                size=1,
                p=weights,
                replace=True,                             # before particles was active_particles
            )[0]

            preferred_particle.preferred = True

            current_particles, tracker_before, tracker_after = particle_gibbs_kernel(
                x["problem"],
                llm,
                prm,
                config,
                n_particles,
                softmax_temp,
                resample_inactive=resample_inactive,
                reference_particle=preferred_particle,
                temperature_annealing=temperature_annealing
            )
            particles_tracker.append(current_particles)
            current_timestep += 1
            particle_intermediate_storage.append([copy.deepcopy(current_particles)])

    # # Create output directory if it doesn't exist
    os.makedirs(config.output_dir, exist_ok=True)

    save_path = os.path.join(config.output_dir, f"{question_id}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(particles_tracker, f)

    intermediate_dir = os.path.join(config.output_dir, "intermediate")
    os.makedirs(intermediate_dir, exist_ok=True)
    save_path_intermediate = os.path.join(intermediate_dir, f"{question_id}_intermediate.pkl")
    with open(save_path_intermediate, "wb") as f:
        pickle.dump(particle_intermediate_storage, f)
        
    #with open(save_path.replace(".pkl", "_before.pkl"), "wb") as f:
    #    pickle.dump(tracker_before, f)
    
    #with open(save_path.replace(".pkl", "_after.pkl"), "wb") as f:
    #    pickle.dump(tracker_after, f)

    logger.info(f"Saved particles to: {save_path}")

    return x
#--------------------------------------------------------------------------------

def particle_gibbs_kernel(
    question,
    llm,
    prm,
    config,
    n_particles,
    softmax_temp,
    resample_inactive,
    reference_particle=None,
    temperature_annealing=(),
    llm_sampling_temp=0.8,
):
    """
    Implements particle Gibbs sampling for response generation.

    Args:
        question: The input question/prompt
        llm: Language model instance
        prm: Parameter object containing reward model
        config: Configuration for LLM
        n_particles: Number of particles to maintain
        resample_inactive: Whether to resample inactive particles
    Returns:
        List of trajectories and their scores
    """
    logger.info("Starting Particle Gibbs sampling...")
    logger.info(f"Particles: {n_particles}")
    logger.info(f"Resample inactive: {resample_inactive}")
    logger.info(f"LLM sampling temperature: {llm_sampling_temp}")
    logger.info(f"Softmax temperature: {softmax_temp}")
    logger.info(f"Temperature annealing: {temperature_annealing}")

    stepwise_particle_tracker_before = []
    stepwise_particle_tracker_after = []

    # Initialize particles
    if reference_particle is None:
        particles = [Particle(temperature=llm_sampling_temp) for _ in range(n_particles)]
    else:
        particles = [Particle(temperature=llm_sampling_temp) for _ in range(n_particles - 1)]

    # print(f"Initialized {n_particles} particles.")

    # Initial step for all particles
    for idx, particle in enumerate(particles):
        response, stop = take_a_step(question, llm, config, first=True, temperature=llm_sampling_temp)
        reward = prm.score([question], [[response]])[-1][-1][-1]
        particle.add_step(response, reward, stop)
        # print(
        #     f"Particle {idx}: Initial response: '{response}', Reward: {reward}, Stop: {stop}"
        # )
#-----------------------------------------------------------
def take_a_step(question, llm, config, steps_so_far=[], first=False, temperature=0.8):
    """
    Generates a response for a single step with a given temperature.
        config: Configuration containing the system prompt.
        steps_so_far (list): Previous steps in the trajectory.
        first (bool): If True, this is the first step (affects prompt construction).
        temperature (float): The sampling temperature for this step.

    Returns:
        tuple: (response_text, stop_reason)
    """
    tokenizer = llm.get_tokenizer()
    system = [
        {
            "role": "system",
            "content": config.system_prompt,
        }
    ]
#system_prompt: str = "Solve the following math problem efficiently and clearly:\n\n-
For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n-
For complex problems (3 steps or more):\nUse this step-by-step format:\n\n
## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n
## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n
...\n\n
Regardless of the approach, always conclude with:\n\n
Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\n
Where [answer] is just the final number or expression that solves the problem."

    sampling_params = SamplingParams(
        temperature=temperature,  # Dynamic temperature
        max_tokens=1024,
        top_p=1.0,
        stop=["\n\n", "<|eot_id|>"],
        stop_token_ids=(
            [151645, 151643]
            if "qwen2" in config.model_path.lower()
            else None),)

    if first:
        prompt = tokenizer.apply_chat_template(
            system + [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,)
    else:
        prompt = tokenizer.apply_chat_template(
            system + [{"role": "user", "content": question}], tokenize=False)
        prompt = prompt + "\n\n".join(steps_so_far) + "\n\n"

    res = llm.generate(prompt, sampling_params)
    response_text = res[0].outputs[0].text
    response_tokens = res[0].outputs[0].token_ids

    if tokenizer.eos_token_id in response_tokens:
        stop_reason = "EOS"
    else:
        stop_reason = "END OF STEP"

    return response_text, stop_reason
#-----------------------------------------------------------

    # print("particles: ", particles)
    # print(particles[0].trajectory)
    # print("----------------" * 20)
    # # print(particles[1].trajectory)
    # print("----------------" * 20)
    # print(particles[2].trajectory)

    stepwise_particle_tracker_before.append([p.deepcopy() for p in particles])

    step = 1

    while any(particle.is_active() for particle in particles):
        if resample_inactive:
            rewards = [particle.get_last_reward() for particle in particles]

            if reference_particle is not None:
                if step >= len(reference_particle.rewards):
                    rewards.append(reference_particle.rewards[-1])
                else:
                    rewards.append(reference_particle.rewards[step])

            logits = [inverse_sigmoid(r) for r in rewards]
            logits = np.array(logits)
            
            if temperature_annealing:
                softmax_temp = temperature_linear_annealing(
                    starting_temp=temperature_annealing[0],
                    ending_temp=temperature_annealing[1],
                    total_steps=temperature_annealing[2],
                    current_step=step,
                )

            logger.info(f"Softmax temperature: {softmax_temp}")
            weights = softmax(logits / softmax_temp)


            # Sample new particles based on weights
            if reference_particle is None:
                sampled_particles = np.random.choice(
                    particles,
                    size=len(particles),
                    p=weights,
                    replace=True,  # before particles was active_particles
                )
            else:
                sampled_particles = np.random.choice(
                    particles + [reference_particle],
                    size=len(particles),
                    p=weights,
                    replace=True,  # before particles was active_particles
                )

            # TODO: change this to allow inactive particles to be sampled (bc perhaps the score of an incomplete still-evolving particle is higher than that of a completed particle)
            # print(
            #    f"Sampled particle indices: {[particles.index(p) for p in sampled_particles+[reference_particle]]}"
            # )
            particles = [
                particle.deepcopy(numSteps=step) for particle in sampled_particles
            ]

        else:
            # Check active particles
            active_particles = [
                particle for particle in particles if particle.is_active()
            ]
            # print(f"Active particles: {len(active_particles)} / {n_particles}")

            # Calculate rewards and weights
            rewards = [particle.get_last_reward() for particle in active_particles]

            if reference_particle is not None:
                if step >= len(reference_particle.rewards):
                    rewards.append(reference_particle.rewards[-1])
                else:
                    rewards.append(reference_particle.rewards[step])

            logits = [inverse_sigmoid(r) for r in rewards]
            logits = np.array(logits)

            if temperature_annealing:
                softmax_temp = temperature_linear_annealing(
                    starting_temp=temperature_annealing[0],
                    ending_temp=temperature_annealing[1],
                    total_steps=temperature_annealing[2],
                    current_step=step,
                )

            logger.info(f"Softmax temperature: {softmax_temp}")
            weights = softmax(logits / softmax_temp)

            # print(f"Rewards of active particles: {rewards}")
            # print(f"Logits (inverse sigmoid): {logits}")
            # print(f"Weights (softmax): {weights}")

            # Sample new particles based on weights
            if reference_particle is None:
                print("len(particles)", len(particles))
                print("len(weights)", len(weights))
                sampled_particles = np.random.choice(
                    active_particles,
                    size=len(active_particles),
                    p=weights,
                    replace=True,  # before particles was active_particles
                )
            else:
                sampled_particles = np.random.choice(
                    active_particles + [reference_particle],
                    size=len(active_particles),
                    p=weights,
                    replace=True,  # before particles was active_particles
                )

            # print(
            #    f"Sampled particle indices: {[particles.index(p) for p in sampled_particles]}"
            # )
            particles = [
                particle.deepcopy()
                for particle in particles
                if not particle.is_active()
            ]

            for i in range(len(sampled_particles)):
                particles.append(sampled_particles[i].deepcopy(numSteps=step))

        stepwise_particle_tracker_after.append([p.deepcopy() for p in particles])

        # Take a step for each sampled particle
        for idx, particle in enumerate(particles):
            if not particle.is_active():
                continue
            response, stop = take_a_step(
                question, llm, config, first=False, steps_so_far=particle.trajectory
            )
            # print("RESPONSE: ", response)
            #print("SELF . TRAJECTORY: ", particle.trajectory)
            response_to_pass_for_score = "\n\n".join(particle.trajectory) + "\n\n" + response
            #print("RESPONSE TO PASS FOR SCORE: ", response_to_pass_for_score)
            reward = prm.score([question], [[response_to_pass_for_score]])[-1][-1][-1]
            #print("HIHIHIIHI CHECK THIS response: ", response)
            particle.add_step(response, reward, stop)
            # print(
            #     f"Particle {particles.index(particle)}: New response: '{response}', Reward: {reward}, Stop: {stop}"
            # )

        step = step + 1
        stepwise_particle_tracker_before.append([p.deepcopy() for p in particles])

    if reference_particle is None:
        return particles, stepwise_particle_tracker_before, stepwise_particle_tracker_after 
    else:
        return particles + [reference_particle], stepwise_particle_tracker_before, stepwise_particle_tracker_after
def particle_gibbs_batch(
    x,
    config,
    llm,
    prm,
    total_timesteps=1,
    n_particles=4,
    resample_inactive=True,
    softmax_temp=1.0,
    temperature_annealing=(),
    llm_sampling_temp=0.8,
):
    
    particle_intermediate_storage = []
    if isinstance(x["unique_id"], int):
        question_id = x["unique_id"]
    else:
        question_id = x["unique_id"].replace("/", "_").strip(".json")
    logger.info(f"Processing question: {question_id}")
    particles_tracker = []
    current_timestep = 1
    current_particles, tracker_before, tracker_after = particle_gibbs_kernel(
                            x["problem"], 
                            llm, 
                            prm, 
                            config, 
                            n_particles, 
                            resample_inactive=resample_inactive,
                            softmax_temp=softmax_temp,
                            temperature_annealing=temperature_annealing,
                            llm_sampling_temp=llm_sampling_temp,
    )
    particles_tracker.append(current_particles)
    particle_intermediate_storage.append([copy.deepcopy(current_particles)])

    # compute_budget_used = sum([len(p.trajectory) for p in current_particles])
    if total_timesteps > 1:
        while current_timestep < total_timesteps:
            # preferred_particle = current_particles[
            #     np.argmax([p.rewards[-1] for p in current_particles])
            # ]
            rewards = [particle.get_last_reward() for particle in current_particles]

            logits = [inverse_sigmoid(r) for r in rewards]
            logits = np.array(logits)

            if temperature_annealing:
                softmax_temp = temperature_linear_annealing(
                    starting_temp=temperature_annealing[0],
                    ending_temp=temperature_annealing[1],
                    total_steps=temperature_annealing[2],
                    current_step=current_timestep,
                )

            logger.info(f"Softmax temperature: {softmax_temp}")
            weights = softmax(logits / softmax_temp)

            preferred_particle = np.random.choice(
                current_particles,
                size=1,
                p=weights,
                replace=True,                             # before particles was active_particles
            )[0]

            preferred_particle.preferred = True

            current_particles, tracker_before, tracker_after = particle_gibbs_kernel(
                x["problem"],
                llm,
                prm,
                config,
                n_particles,
                softmax_temp,
                resample_inactive=resample_inactive,
                reference_particle=preferred_particle,
                temperature_annealing=temperature_annealing
            )
            particles_tracker.append(current_particles)
            current_timestep += 1
            particle_intermediate_storage.append([copy.deepcopy(current_particles)])

    # # Create output directory if it doesn't exist
    os.makedirs(config.output_dir, exist_ok=True)

    save_path = os.path.join(config.output_dir, f"{question_id}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(particles_tracker, f)

    intermediate_dir = os.path.join(config.output_dir, "intermediate")
    os.makedirs(intermediate_dir, exist_ok=True)
    save_path_intermediate = os.path.join(intermediate_dir, f"{question_id}_intermediate.pkl")
    with open(save_path_intermediate, "wb") as f:
        pickle.dump(particle_intermediate_storage, f)
        
    #with open(save_path.replace(".pkl", "_before.pkl"), "wb") as f:
    #    pickle.dump(tracker_before, f)
    
    #with open(save_path.replace(".pkl", "_after.pkl"), "wb") as f:
    #    pickle.dump(tracker_after, f)

    logger.info(f"Saved particles to: {save_path}")

    return x

```
```python
물론입니다! 이전에 설명한 POMCP (Partially Observable Monte Carlo Planning) 알고리즘의 핵심 개념을 바탕으로, 간단한 Python 예제와 함께 원본 논문 정보를 제공해드리겠습니다. 


---

🧠 POMCP 간단한 Python 예제

아래는 POMCP의 핵심 아이디어를 간단하게 구현한 Python 예제입니다. 

import random
import math
from collections import defaultdict

class POMCPNode:
    def __init__(self):
        self.children = dict()  # action: POMCPNode
        self.visits = 0
        self.value = 0.0

class POMCP:
    def __init__(self, simulator, ucb_constant=1.0, num_simulations=1000):
        self.simulator = simulator  # 환경 시뮬레이터
        self.ucb_constant = ucb_constant
        self.num_simulations = num_simulations
        self.root = POMCPNode()

    def search(self, belief):
        for _ in range(self.num_simulations):
            state = random.choice(belief)  # 입자에서 상태 샘플링
            self.simulate(state, self.root, depth=0)
        return max(self.root.children.items(), key=lambda item: item[1].value / item[1].visits)[0]

    def simulate(self, state, node, depth):
        if depth > self.simulator.max_depth:
            return 0.0
        if not node.children:
            for action in self.simulator.get_actions(state):
                node.children[action] = POMCPNode()
            return self.rollout(state, depth)
        # UCT를 사용하여 행동 선택
        total_visits = sum(child.visits for child in node.children.values())
        log_total = math.log(total_visits + 1)
        best_score = -float('inf')
        best_action = None
        for action, child in node.children.items():
            if child.visits == 0:
                score = float('inf')
            else:
                score = (child.value / child.visits) + self.ucb_constant * math.sqrt(log_total / child.visits)
            if score > best_score:
                best_score = score
                best_action = action
        next_state, observation, reward = self.simulator.step(state, best_action)
        if best_action not in node.children:
            node.children[best_action] = POMCPNode()
        child_node = node.children[best_action]
        q = reward + self.simulate(next_state, child_node, depth + 1)
        child_node.visits += 1
        child_node.value += q
        return q

    def rollout(self, state, depth):
        if depth > self.simulator.max_depth:
            return 0.0
        action = random.choice(self.simulator.get_actions(state))
        next_state, observation, reward = self.simulator.step(state, action)
        return reward + self.rollout(next_state, depth + 1)

설명:

POMCPNode: 각 노드를 나타내며, 자식 노드, 방문 횟수, 가치 값을 저장합니다.

POMCP: POMCP 알고리즘의 주요 클래스입니다.

search: 주어진 belief(입자 집합)에서 최적의 행동을 찾습니다.

simulate: 시뮬레이션을 통해 노드의 가치를 업데이트합니다.

rollout: 임의의 정책을 사용하여 시뮬레이션을 진행합니다.



주의사항:

이 코드는 POMCP의 핵심 아이디어를 간단하게 구현한 것으로, 실제 환경에서는 더 복잡한 요소들이 필요합니다.

simulator는 환경의 동작을 정의하는 클래스이며, get_actions, step, max_depth 등의 메서드를 구현해야 합니다.



---

📄 POMCP 원본 논문 정보

POMCP 알고리즘은 다음 논문에서 처음 소개되었습니다: 

제목: Monte-Carlo Planning in Large POMDPs
저자: David Silver, Joel Veness
학회: Advances in Neural Information Processing Systems (NeurIPS) 2010 

논문 링크:

NeurIPS 공식 링크: 

PDF 다운로드: 


이 논문에서는 POMCP 알고리즘이 어떻게 대규모 POMDP 문제를 효율적으로 해결할 수 있는지에 대해 자세히 설명하고 있습니다. 


---

추가적으로, POMCP의 다양한 변형 및 응용에 대한 연구들도 진행되고 있습니다. 예를 들어, POMCP-PM(POMCP with Path Modification)은 가치 업데이트 경로를 수정하여 성능을 향상시키는 방법을 제안합니다. 

관련 논문:

A Partially Observable Monte Carlo Planning Algorithm Based on Path Modification




더 자세한 내용이나 특정 부분에 대한 설명이 필요하시면 언제든지 말씀해주세요!


```
