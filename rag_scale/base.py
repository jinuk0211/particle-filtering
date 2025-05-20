
from generator import Generator, load_vLLM_model, generate_with_vLLM_model, llm_proposal, LLM
from prompt import rag_prompt
from cfg import cfg
from huggingface_hub import login
from datasets import load_dataset, Dataset
import torch
import time
from evaluate import run_evaluation
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Subquestion generator script.")
    
    parser.add_argument(
        '--ds',
        type=str,
        default='popqa',
        help='name of data'
    )        
    return parser.parse_args()

#tokenizer, model = load_vLLM_model(cfg.model_ckpt, cfg.seed, cfg.tensor_parallel_size, cfg.half_precision)
model, tokenizer = LLM('qwen2.5')
split = 100
output_list = []
input_list = []

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
  
  with torch.no_grad():
    for row in range(len(df)):
    
        if args.ds == 'gpqa':
          question = ds['train']['Question'][row]
          prompt = ds['train'][row]['Question'] + 'A)' + ds['train'][row]['Correct Answer'].replace('\n','') + ' B)' + ds['train'][row]['Incorrect Answer 1'].replace('\n','') + ' C)' + ds['train'][row]['Incorrect Answer 2'].replace('\n','') + ' D)' + ds['train'][row]['Incorrect Answer 3'].replace('\n','')
          
        elif args.ds == 'popqa':
          question = df['question'][row]
          prompt = question
           
        elif args.ds == 'arc':
          question = df['question'][row]
          prompt = question + "\nA)" + df['choices'][row]['text'][0] + " B)" + df['choices'][row]['text'][1] + " C)" + df['choices'][row]['text'][2] + " D)" + df['choices'][row]['text'][3]
                

        input_list.append(question)
          
        response = llm_proposal(model,tokenizer,prompt)
        print(response)
        final_input =  "Question: " + prompt + "\nAnswer: " + response[0] + '\n\n' + 'Extract only the final answer, without restating the question or explanation. Provide the answer as a short, direct response.'
        only_answer = llm_proposal(model,tokenizer,final_input)[0]
        output_list.append(only_answer)
        print(f'----------f{row} question done-----------')
    torch.cuda.empty_cache()
    print(output_list)
    # evaluator.evaluate(only_answer, ds['train']['Correct Answer'][i])
    run_evaluation(df,input_list,output_list, output_dir='abr/coss39/rag_scale/output/',dataset_name = 'popqa')


