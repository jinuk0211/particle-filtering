from generator import Generator, load_vLLM_model, generate_with_vLLM_model, llm_proposal, LLM
from prompt import rag_prompt
from generator import retriever,retriever2
from cfg import cfg
from huggingface_hub import login
from datasets import load_dataset, Dataset
import torch
import time
from evaluate import run_evaluation
from langchain_community.retrievers import WikipediaRetriever
import pandas as pd
import os, sys
sys.path.insert(0, "../self_rag/retrieval_lm")
from passage_retrieval import Retriever
import argparse


output_list = []
input_list = []
rag = []

def parse_args():
    parser = argparse.ArgumentParser(description="Subquestion generator script.")
    
    parser.add_argument(
        '--retriever',
        type=int,
        default=1,
        help='version'
    )
    parser.add_argument(
        '--subqidx',
        type=int,
        default=None,
        help='if retrieval for subquestion, which subquestion'
    )
    parser.add_argument(
        '--ds',
        type=str,
        default='popqa',
        help='name of data'
    )        
    return parser.parse_args()
def is_valid_text(x):
    if x is None:
        return False
    if pd.isna(x):  
        return False
    x_str = str(x).strip().lower()
    if x_str in ['', 'none', 'nan', 'null', 'n/a', 'na']:
        return False
    return True    
if __name__ == "__main__":

  args = parse_args()
  retriever = Retriever({})
  retriever.setup_retriever_demo("facebook/contriever-msmarco", f"enwiki_2020/filtered{args.retriever}.tsv", f"enwiki_2020/enwiki_2020_contriever{args.retriever}/*",  n_docs=5,   save_or_load_index=False)
  if args.ds == 'gpqa':
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
    df = ds['train'].to_pandas()
  elif args.ds == 'popqa':
    df = pd.read_csv("hf://datasets/akariasai/PopQA/test.tsv", sep="\t")
  elif args.ds == 'arc':
    splits = {'train': 'ARC-Challenge/train-00000-of-00001.parquet', 'test': 'ARC-Challenge/test-00000-of-00001.parquet', 'validation': 'ARC-Challenge/validation-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/allenai/ai2_arc/" + splits["train"])
  elif args.ds == 'musique':
    ds = load_dataset("dgslibisey/MuSiQue")
    
  with torch.no_grad():
    if args.subqidx: #scale rag
      subq_df = pd.read_csv(f'subquestions_{args.ds}.csv')
      for row in range(200): #when index
      #for row in range(split):
          question = subq_df[f"subquestion{args.subqidx}"].iloc[row]
          if is_valid_text(question):
            input_list.append(question)
            print(question)
            retrieved_document = retriever.search_document_demo(question, 1)    
            text = retrieved_document[0]['text']
            rag.append(text)
          else: 
            rag.append('None')
            print('---------------no question existed------------')
      #subq_df[f"text_subquestion{args.subqidx}_{args.retriever}"] = rag
      subq_df[f"text_subquestion{args.subqidx}_{args.retriever}"] = rag  #when index
      subq_df.to_csv(f"subquestions_{args.ds}", index=False, encoding="utf-8")
      
      
    else: #naive rag
      for row in range(len(df)):
      #for row in range(split,len(ds['train'])): #when index
          
          if args.ds == 'gpqa':
            question = ds['train']['Question'][row]
            prompt = ds['train'][row]['Question'] + 'A)' + ds['train'][row]['Correct Answer'].replace('\n','') + ' B)' + ds['train'][row]['Incorrect Answer 1'].replace('\n','') + ' C)' + ds['train'][row]['Incorrect Answer 2'].replace('\n','') + ' D)' + ds['train'][row]['Incorrect Answer 3'].replace('\n','')
          elif args.ds == 'popqa':
            question = df['question'][row]
            prompt = question 
          elif args.ds == 'arc':
            question = df['question'][row]
            prompt = question + "\nA)" + df['choices'][row]['text'][0] + " B)" + df['choices'][row]['text'][1] + " C)" + df['choices'][row]['text'][2] + " D)" + df['choices'][row]['text'][3]    
                    
          print(question)
          input_list.append(question)
          retrieved_document = retriever.search_document_demo(question, 1)    
          text = retrieved_document[0]['text']
  
          rag.append(text)
  
          #rag.append({"index": row, "text": text})
      if os.path.exists(f"retrieved_texts_{args.ds}.csv"): 
        df = pd.read_csv(f"retrieved_texts_{args.ds}.csv")
      else:
        df = pd.DataFrame({
        f"text{args.retriever}": rag})
    
    
      df[f"text{args.retriever}"] = rag #when index

      df.to_csv(f"retrieved_texts_{args.ds}.csv", index=False, encoding="utf-8")        