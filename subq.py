from collections import defaultdict
from evaluator import GPQAEvaluator
from generator import Generator,  LLM
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
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('subq')
logger.setLevel(logging.INFO)
import pickle
import copy
import re


model_name = "Qwen/Qwen2.5-7B-Instruct"
global global_value_model, global_tokenizer
model,tokenizer=LLM(model='qwen2.5')
global_value_model = model       
global_tokenizer = tokenizer
evaluator = GPQAEvaluator()
print('llm done')

config = Config()
config.output_dir = 'math'
config.model_path = "meta-llama/Llama-3.2-1B-Instruct"
config.prm_path = "Qwen/Qwen2.5-Math-PRM-7B"
llm_sampling_temp = 0.8
#n_particles = 2
reference_particle=None
temperature_annealing=()
softmax_temp=1.0
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
    parser.add_argument(
        '--particles',
        type=int,
        default=8,
        help='name of data'
    )                    
    return parser.parse_args()
from generator import concat_subqs_and_subas
particle_list = []
from pg import Particle, inverse_sigmoid, temperature_linear_annealing, softmax, stop_reason
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
        
    for row in range(2,11):
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
      print(f'question:{question}')

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
      
      
      
      stepwise_particle_tracker_before = []
      stepwise_particle_tracker_after = []
  
      # Initialize particles
      if reference_particle is None:
          particles = [Particle(temperature=llm_sampling_temp) for _ in range(args.particles)]
      else:
          particles = [Particle(temperature=llm_sampling_temp) for _ in range(args.particles - 1)]
  
      print(f"Initialized {args.particles} particles.")
      
      #responses, stops = take_a_step_for_batch(question, llm, config, first=True, temperature=llm_sampling_temp, n_particles=len(particles))
      subquestions, subqids = generator.subq(question,subq_list,suba_list,num_subq=5)
      for i in subquestions:
        value = probability_subquestion_question(question, i) # probablistic score
        value_list.append(value)
      
      top_indices = np.argsort(value_list)[::-1][:1]
      top_subq = [subquestions[i] for i in top_indices] #top3 subquestions
      top_subq = top_subq[0]
      top_subq = re.split(r'[sS]ubanswer', top_subq)[0]
      if 'subquestion' or 'Subquestion' in top_subq:
        top_subq = top_subq.replace('subquestion','')
        top_subq = top_subq.replace('Subquestion','')
      print(f'first_subq:{top_subq}')
       
      
      for particle in particles:    
        particle.subq_list.append(top_subq)
          
      for idx, particle in enumerate(particles):
          #response, stop = take_a_step(question, llm, config, first=True, temperature=llm_sampling_temp)
          subanswers, subaids = generator.suba(question,particle.subq_list,particle.suba_list,num_suba = 1)
          stop = stop_reason(tokenizer, subaids[0].tolist(),subanswers[0])
          print(f'first_subq_generation_particle_{idx}_stop:{stop}')
          reward = prm.score([question], [[subanswers[0]]])[-1][-1][-1]
          suba = subanswers[0]
          suba = re.split(r'[sS]ubquestion', suba)[0].strip()
          particle.suba_list.append(suba)
          #particle.add_step(response, reward, stop)
          particle.add_step(suba,reward, stop)
   
      
  
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
          

            
          for idx, particle in enumerate(particles):
              
              if not particle.is_active():
                  continue      
              value_list = []                  
              subquestions, subqids = generator.subq(question,particle.subq_list,particle.suba_list,num_subq=5)
              for i in subquestions:
                value = probability_subquestion_question(question, i) # probablistic score
                value_list.append(value)
                
              top_indices = np.argsort(value_list)[::-1][:1]
              top_subq = [subquestions[i] for i in top_indices] #top3 subquestions
              top_subq = top_subq[0]
              top_subq = re.split(r'[sS]ubanswer', top_subq)[0].strip()
              if 'subquestion' or 'Subquestion' in top_subq:
                suba = suba.replace('subquestion','')
                suba = suba.replace('Subquestion','')
              print('--------------')
              print(f'particle{idx}_subq:{top_subq}')
              
              particle.subq_list.append(top_subq)
                           
            
            
                  
              #response, stop = take_a_step(question, llm, config, first=False, steps_so_far=particle.trajectory)
              subanswers, subaids = generator.suba(question,particle.subq_list,particle.suba_list,num_suba = 1)
              stop = stop_reason(tokenizer,subaids[0].tolist(),subanswers[0])
              suba = subanswers[0]
              suba = re.split(r'[sS]ubquestion', suba)[0].strip()
              if 'subanswer' or 'Subanswer' in suba:
                suba = suba.replace('subanswer','')
                suba = suba.replace('Subanswer','')
              particle.suba_list.append(suba)
              response_to_pass_for_score = "\n\n".join(particle.trajectory) + "\n\n" + suba
              print(f'particle{idx}_suba:{suba}\n')
              print('--------------')
              reward = prm.score([question], [[response_to_pass_for_score]])[-1][-1][-1]
              
              particle.add_step(subanswers[0], reward, stop)
              #print(f'particle trajectory:{particle.trajectory}')
              #if suba == '':
                #particle.active = False
          torch.cuda.empty_cache()
 
          step = step + 1
          stepwise_particle_tracker_before.append([p.deepcopy() for p in particles])
      if reference_particle is None:
          current_particles = particles
          tracker_before = stepwise_particle_tracker_before
          tracker_after =  stepwise_particle_tracker_after
        #    particles, stepwise_particle_tracker_before, stepwise_particle_tracker_after 
      else:
          current_particles = particles + [reference_particle] 
      #def gibbs kernel end
      particles_tracker.append(current_particles)
      particle_intermediate_storage.append([copy.deepcopy(current_particles)])
  
      # compute_budget_used = sum([len(p.trajectory) for p in current_particles])
      
      # # Create output directory if it doesn't exist
      os.makedirs(config.output_dir, exist_ok=True)
  
      save_path = os.path.join(config.output_dir, f"question{row}.pkl")
      with open(save_path, "wb") as f:
          pickle.dump(particles_tracker, f)
  
      intermediate_dir = os.path.join(config.output_dir, f"question{row}_intermediate.pkl")
      
      with open(intermediate_dir, "wb") as f:
          pickle.dump(particle_intermediate_storage, f)
          
  
      logger.info(f"Saved particles to: {save_path}")
  
      rewards = [x.rewards[-1] for x in current_particles]
      best_particle = current_particles[np.argmax(rewards)]
      print(f'question{row}_best_particle:{best_particle.trajectory}')
      particle_list.append(best_particle.trajectory)
    df = pd.DataFrame(particle_list, columns=['particle'])
    df.to_csv(f'output_{args.ds}_20100.csv', index=False, encoding="utf-8")            
    


        #if 'Now we can' in subanswers[0]:
          #print('done')
          #break
      #output = generator.final_output(question,subq_list,F)
      #output_list.append(output)
    #result = pd.DataFrame(output)
    #result.to_csv(f'output_{args.ds}.csv', index=False, encoding="utf-8")

        
def concat_subqs_and_subas(question_list, answer_list):
    result = ""

    # 질문 리스트가 비어 있으면 바로 반환
    if not question_list:
        
        return result, 1

    min_len = min(len(question_list), len(answer_list))

    # 질문-답변 쌍 처리
    for i in range(min_len):
        result += f"subquestion {i+1}: {question_list[i]}\n"
        #result += f"subanswer {i+1}: The answer is: {answer_list[i]}.\n"
        result += f"subanswer {i+1}: {answer_list[i]}.\n"
    # 질문이 하나 더 많을 경우 마지막 질문만 출력
    if len(question_list) > len(answer_list):
        result += f"subquestion {len(question_list)}: {question_list[-1]}\n"

    next_subquestion_id = len(question_list) + 1
    return result, next_subquestion_id
