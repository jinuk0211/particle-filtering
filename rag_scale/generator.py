import os, sys
# sys.path.insert(0, "../self_rag/retrieval_lm")
# from passage_retrieval import Retriever

# retriever = Retriever({})
#retriever.setup_retriever_demo("facebook/contriever-msmarco", "enwiki_2020/filtered.tsv", "enwiki_2020/enwiki_2020_contriever/*",  n_docs=5, save_or_load_index=False)
# retriever2 = Retriever({})
#retriever2.setup_retriever_demo("facebook/contriever-msmarco", "enwiki_2020/filtered1.tsv", "enwiki_2020/enwiki_2020_contriever2/*",  n_docs=5, save_or_load_index=False)
# retriever3 = Retriever({})
#retriever3.setup_retriever_demo("facebook/contriever-msmarco", "enwiki_2020/filtered2.tsv", "enwiki_2020/enwiki_2020_contriever3/*",  n_docs=5, save_or_load_index=False)
#:nwiki_2020_intro_only/enwiki_dec_2020_contriever_intro/*"
# workspace/self-rag/retrieval_lm/enwiki_2020_intro_only/enwiki_2020_dec_intro_only.jsonl
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
import math
from evaluator import Evaluator
from collections import defaultdict
from typing import List, Dict, Tuple
import re
from prompt import rag_prompt
def qwen(model):
     model_dict = {}
     if model == 'qwen3':
         print('init llm model')       
         # model_name = "Qwen/Qwen2.5-7B-Instruct"
         model_name = "Qwen/Qwen3-8B"

         model = AutoModelForCausalLM.from_pretrained(
             model_name,
             torch_dtype="auto",
             device_map="auto"
         )
         tokenizer = AutoTokenizer.from_pretrained(model_name)
        
         return model, tokenizer
     if model == 'qwen2.5':
         print('init llm model')       
         model_name = "Qwen/Qwen2.5-7B-Instruct"
         model = AutoModelForCausalLM.from_pretrained(
             model_name,
             torch_dtype="auto",
             device_map="auto"
         )
         tokenizer = AutoTokenizer.from_pretrained(model_name)

         return model, tokenizer 

def llm_proposal_score(model=None,tokenizer=None,messages=None,temperature='0.8',model_name='qwen',n=1):
    if model_name =='qwen':
          prompt = [
                {"role": "system", "content": "You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems. Given a user question and the retrieved context passages, assess the overall quality"},
              {"role": "user", "content": messages}
          ]            
          
          text = tokenizer.apply_chat_template(
              prompt, tokenize=False,
              add_generation_prompt=True)
          
          model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
          generated_ids = model.generate(
              **model_inputs, max_new_tokens=512,temperature=0.8)
          
          generated_ids = [
              output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

          response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
          return response
                            
              
    
def llm_proposal(model=None,tokenizer=None,messages=None,temperature='0.8',model_name='qwen',n=1,max_new_tokens=512,stop_tokens = None):
    if model_name =='qwen':
        if isinstance(messages, str):
            prompt = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": messages}
            ]            
            output_list = []
            id_list = []
            text = tokenizer.apply_chat_template(
                prompt, tokenize=False,
                add_generation_prompt=True)
            
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            for i in range(n):
                generated_ids = model.generate(
                    **model_inputs, max_new_tokens=max_new_tokens,temperature=0.8,eos_token_id =stop_tokens)
                
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    
                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                output_list.append(response)
                id_list.append(generated_ids)
            return output_list ,id_list
        elif isinstance(messages, list):
            output_list = []
            id_list = []
            for i in range(len(messages)):
                prompt = [
                    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {"role": "user", "content": messages[i]}
                ]                          
                text = tokenizer.apply_chat_template(
                    prompt, tokenize=False,
                    add_generation_prompt=True)
            
                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
                subanswer_list = []
                subid_list = []
                for k in range(n):
                    generated_ids = model.generate(
                        **model_inputs, max_new_tokens=max_new_tokens,
                        #temperature=0.8
                        )
                    
                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        
                    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    subanswer_list.append(response)
                    subid_list.append(generated_ids)
                output_list.append(subanswer_list)
                id_list.append(subid_list)
            return output_list, id_list            
            
def load_vLLM_model(model_ckpt, seed, tensor_parallel_size=1, half_precision=True, max_num_seqs=256):
    import os
    os.environ['VLLM_USE_V1'] = '0'
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    llm = LLM(
        model=model_ckpt,
        # tensor_parallel_size=tensor_parallel_size,
    )

    return tokenizer, llm


def generate_with_vLLM_model(
    model,
    input,
    temperature=0.8,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1,
    n=1,
    max_tokens=256,
    logprobs=1,
    stop=[],
):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        n=n,
        logprobs=logprobs,
        max_tokens=max_tokens,
        stop=stop,
    )

    output = model.generate(input, sampling_params, use_tqdm=False)

    return output

class GeneratorError(Exception):
    def __init__(self, source, io_input, io_output_list) -> None:
        super().__init__()

        self.source = source
        self.io_input = io_input
        self.io_output_list = io_output_list
class IO_System:
    """Input/Output system"""

    def __init__(self, args, tokenizer, model) -> None:
        self.api = args.api
        if self.api == "together":
            assert tokenizer is None and model is None
        elif self.api == "gpt3.5-turbo":
            assert tokenizer is None and isinstance(model, str)
        self.model_ckpt = args.model_ckpt
        self.temperature = args.temperature
        self.top_k = args.top_k
        self.top_p = args.top_p
        self.tokenizer = tokenizer
        self.model = model

        self.call_counter = 0
        self.token_counter = 0

    def generate(self, model_input, max_tokens: int, num_return: int, stop_tokens):
        if self.api == "gpt-4o" and len(stop_tokens) > 4:
            stop_tokens = stop_tokens[:4]

        if isinstance(model_input, str):
            if self.api == "vllm":
                vllm_response = generate_with_vLLM_model(
                    self.model,
                    input=model_input,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    n=num_return,
                    max_tokens=max_tokens,
                    stop=stop_tokens,
                )
                io_output_list = [o.text for o in vllm_response[0].outputs]
                response_token = vllm_response[0].outputs[0].token_ids
                if self.tokenizer.eos_token_id in response_token:
                    stop_reason = "EOS"
                else:
                    stop_reason = "END OF STEP"
                self.call_counter += 1
                self.token_counter += sum([len(o.token_ids) for o in vllm_response[0].outputs])
                
            elif self.api == 'huggingface':
                io_output_list,ids = llm_proposal(self.model, self.tokenizer, model_input, n=num_return,max_new_tokens=max_tokens,stop_tokens=stop_tokens)
                
                self.call_counter += 1
                # self.token_counter += sum([len(o.token_ids) for o in llm_response[0].outputs])
                
        
            elif self.api == "debug":
                io_output_list = ["Debug: The answer is generated with debug mode, 233." for _ in range(num_return)]
            else:
                raise NotImplementedError(f"API {self.api} is not implemented.")
        elif isinstance(model_input, list):
            if self.api == "vllm":
                vllm_response = generate_with_vLLM_model(
                    self.model,
                    input=model_input,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    n=num_return,
                    max_tokens=max_tokens,
                    stop=stop_tokens,
                )
                io_output_list = [
                    [o.text for o in resp_to_single_input.outputs] for resp_to_single_input in vllm_response
                ]
                self.call_counter += 1
                self.token_counter += sum(
                    [
                        sum([len(o.token_ids) for o in resp_to_single_input.outputs])
                        for resp_to_single_input in vllm_response
                    ]
                )
            elif self.api == 'huggingface':
                io_output_list,ids = llm_proposal(self.model, self.tokenizer, model_input,n=num_return)
                self.call_counter += num_return * len(model_input)
            
        
            elif self.api == "debug":
                io_output_list = [
                    ["Debug: The answer is generated with debug mode, 233." for _ in range(num_return)]
                    for _ in model_input
                ]
            else:
                raise NotImplementedError(f"API {self.api} is not implemented.")

        return io_output_list, stop_reason
        
def concat_subqs_and_subas(question_list, answer_list):
    result = ""

    # 질문 리스트가 비어 있으면 바로 반환
    if not question_list:
        
        return result, 1

    min_len = min(len(question_list), len(answer_list))

    # 질문-답변 쌍 처리
    for i in range(min_len):
        result += f"subquestion {i+1}: {question_list[i]}\n"
        result += f"subanswer {i+1}: The answer is: {answer_list[i]}.\n"

    # 질문이 하나 더 많을 경우 마지막 질문만 출력
    if len(question_list) > len(answer_list):
        result += f"subquestion {len(question_list)}: {question_list[-1]}\n"

    next_subquestion_id = len(question_list) + 1
    return result, next_subquestion_id
    
class Generator:
    """Generator generates children nodes"""

    def __init__(self, args, tokenizer, model, evaluator: Evaluator) -> None:
        self.io = IO_System(args, tokenizer, model)
        self.evaluator = evaluator
        if not args.disable_rag:
            self.retriever = Retriever()
            self.retriever.regist_io_system(self.io)

        self.num_subquestions = args.num_subquestions

        self.num_votes = args.num_votes
        self.max_tokens = args.max_tokens
        self.enable_potential_score = args.enable_potential_score

        self.mcts_num_last_votes = args.mcts_num_last_votes


        # if self.solution_trace is None:  # root
        #     self.solution_trace: Dict[int, Dict[str, str]] = {0: {"user_question": user_question, "ost_step": {}}}
        # if node_type is Node_Type.REPHRASED_USER_QUESTION:
        #     self.solution_trace[0]["user_question"] = rephrased_user_question

    def _extract_from_cache(self, subquestion_list: List[str]):
        high_score_questions = []
        selected_answers = []
        values = []
        low_score_questions = []
        low_score_values = []
        low_score_answers_list = []
        unmatched_questions = []

        for subquestion in subquestion_list:
            best_match = process.extractOne(subquestion, self.reasoning_cache.keys(), scorer=fuzz.ratio)

            if best_match:
                best_question, best_score = best_match[0], best_match[1]
                similarity = best_score / 100
                cache_entry = self.reasoning_cache[best_question]
                score = cache_entry["score"]
                if similarity == 1:
                    if score >= 0.9:
                        high_score_questions.append(best_question)
                        selected_answers.append(cache_entry["selected_answer"])
                        values.append(score)
                    else:
                        low_score_questions.append(best_question)
                        low_score_values.append(score)
                        low_score_answers_list.append(cache_entry["answer_list"])
                else:
                    unmatched_questions.append(subquestion)
            else:
                unmatched_questions.append(subquestion)

        return {
            "high_score_questions": high_score_questions,
            "selected_answers": selected_answers,  # most likely answer corresponding to each subquestion
            "values": values,
            "low_score_questions": low_score_questions,
            "low_score_values": low_score_values,
            "low_score_answers_list": low_score_answers_list,
            "unmatched_questions": unmatched_questions,
        }


    def _get_most_likely_answer(self, io_output_list: List[str]) -> Tuple[str, float]:
        assert len(io_output_list) > 0

        if len(io_output_list) == 1:
            most_confident_answer_full_completion = io_output_list[0]
            confidence = 1
        else:
            _, most_confident_answer_full_completion, _, confidence = self.evaluator.find_most_confident_answer(
                io_output_list
            )
            assert confidence > 0

        return most_confident_answer_full_completion, confidence

    def _fewshot_cot_answer_question(self, question: str, paraphrased: bool, num_return: int, hint: str = None):
        # fewshot_cot_prompt = self.fewshot_cot_prompt if not paraphrased else self.fewshot_cot_prompt_rephrased
        fewshot_cot_prompt ='''### Instruction:
        There are 15 trees in the grove. Grove workers will plant trees in the grove today.
        After they are done, there will be 21 trees. How many trees did the grove workers plant today?

        ### Response:
        Let's think step by step. There are 15 trees originally.
        Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is: 6.'''

        question += "\n\n" + hint if hint is not None else ""
        # io_input = self.fewshot_cot_config["prompt_template"].format(examples=fewshot_cot_prompt, instruction=question)

        io_input = '''A chat between a curious user and an AI assistant.
        The assistant gives step-by-step solutions to the user's questions.
        You are presented with observations or results related to a phenomenon.
        Based on the information provided, infer the possible reasons or explanations for the observed outcomes.
        In the end of assistant's response, a final answer must be given in the format of \"The answer is: <ANSWER>.\",
        where <ANSWER> should only be \"A\", \"B\", \"C\" or \"D\" without any description.\n\n{examples}\n\n
        # ### Instruction:\n{instruction}\n\n### Response:\n\nPlease answer it in a complete sentence",'''.format(examples=fewshot_cot_prompt, instruction=question)
        io_output_list = self.io.generate(
            io_input,
            num_return=num_return,
            max_tokens=self.max_tokens,
            stop_tokens=self.fewshot_cot_config["stop_tokens"],
        )
        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]  #! cleaning
        return io_input, cleaned_io_output_list

    # def _fewshot_cot_answer_question_with_external_knowledge(self, question: str, external_knowledge: str, paraphrased: bool, num_return: int, hint: str = None):
    #     fewshot_cot_prompt = self.fewshot_cot_prompt if not paraphrased else self.fewshot_cot_prompt_rephrased
    #     question += "\n\n" + hint if hint is not None else ""
    #     io_input = self.fewshot_cot_config["prompt_template"].format(examples=fewshot_cot_prompt, instruction=question)
    #     io_input += (
    #            f"Context: {external_knowledge}"
    #         )
    #     io_output_list = self.io.generate(
    #         io_input,
    #         num_return=num_return,
    #         max_tokens=self.max_tokens,
    #         stop_tokens=self.fewshot_cot_config["stop_tokens"],
    #     )
    #     cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]  #! cleaning
    #     return io_input, cleaned_io_output_list

    def generate_direct_answers(self, user_question: str, paraphrased: bool, hint: str):
        direct_answer_list, value_list = [], []

        #! few shot cot
        num_return = self.mcts_num_last_votes
        io_input, cleaned_io_output_list = self._fewshot_cot_answer_question(
            question=user_question, paraphrased=paraphrased, num_return=num_return, hint=hint
        )

        try:
            most_likely_answer, likelihood = self._get_most_likely_answer(cleaned_io_output_list)
        except Exception as e:
            raise GeneratorError(
                source="generate direct answer from: few shot cot",
                io_input=io_input,
                io_output_list=cleaned_io_output_list,
            )

        direct_answer_list.append(most_likely_answer)
        value_list.append(likelihood)

        return direct_answer_list, value_list
    def _get_most_likely_answer(self, io_output_list: List[str]) -> Tuple[str, float]:
        assert len(io_output_list) > 0

        if len(io_output_list) == 1:
            most_confident_answer_full_completion = io_output_list[0]
            confidence = 1
        else:
            _, most_confident_answer_full_completion, _, confidence = self.evaluator.find_most_confident_answer(
                io_output_list
            )
            assert confidence > 0

        return most_confident_answer_full_completion, confidence
    def find_most_confident_answer(self, completions: List[str], prior_weights: List[float] = None):
        """Returns the most confident answer, its completion, its id in the input list, and its confidence."""
        if completions is None or len(completions) == 0:
            return None, None, None, None

        answer2completions = defaultdict(list)
        answer2ids = defaultdict(list)
        for id, c in enumerate(completions):
            try:
                model_answer = self.extract_answer_from_model_completion(c)
                has_existed = False
                for existing_answer in answer2completions.keys():
                    if self.check_answers_equiv(model_answer, existing_answer):
                        assert not has_existed
                        has_existed = True
                        answer2completions[existing_answer].append(c)
                        answer2ids[existing_answer].append(id)
                if not has_existed:
                    answer2completions[model_answer].append(c)
                    answer2ids[model_answer].append(id)
            except:
                pass

        assert len(answer2completions.keys()) > 0, "There are no valid completions."
        if prior_weights is not None:
            assert len(completions) == len(prior_weights)
            completion2count = {}
            for answer, answer_completions in answer2completions.items():
                count = len(answer_completions)
                for answer_completion in answer_completions:
                    completion2count[answer_completion] = count

            completion2score = {}
            for id, (completion, count) in enumerate(completion2count.items()):
                prior_weight = prior_weights[id]
                score = prior_weight * (count / len(completions))
                completion2score[completion] = score

            most_confident_completion = max(completion2score.keys(), key=lambda x: completion2score[x])

            return (
                self.extract_answer_from_model_completion(most_confident_completion),
                most_confident_completion,
                completions.index(most_confident_completion),
                completion2score[most_confident_completion],
            )
        else:
            most_confident_answer = max(answer2completions.keys(), key=lambda x: len(answer2completions[x]))
            assert (
                len(answer2completions[most_confident_answer]) > 0
            ), "There are no completions for the most confident answer."
            confidence = len(answer2completions[most_confident_answer]) / len(completions)
            assert confidence > 0
            return (
                most_confident_answer,
                answer2completions[most_confident_answer][0],
                answer2ids[most_confident_answer][0],
                confidence,
            )


    def generate_direct_answers(self, user_question: str, paraphrased: bool, hint: str):
        direct_answer_list, value_list = [], []

        #! few shot cot
        num_return = self.mcts_num_last_votes
        io_input, cleaned_io_output_list = self._fewshot_cot_answer_question(
            question=user_question, paraphrased=paraphrased, num_return=num_return, hint=hint
        )

        try:
            most_likely_answer, likelihood = self._get_most_likely_answer(cleaned_io_output_list)
        except Exception as e:
            raise GeneratorError(
                source="generate direct answer from: few shot cot",
                io_input=io_input,
                io_output_list=cleaned_io_output_list,
            )

        direct_answer_list.append(most_likely_answer)
        value_list.append(likelihood)

        return direct_answer_list, value_list

# (subquestion_list, subanswer_list, value_list, potential_answers_list) = (
#     self.generator.generate_subquestions(
#                     user_question=self.user_question, solution_trace=self.solution_trace, paraphrased=self.paraphrased
#                 ))
    def subq(self, user_question: str, question_list, answer_list,num_subq):
        decompose_prompt = """Given a question, please decompose it into sub-questions. For each sub-question, please answer it in one complete sentence, ending with "The answer is ". When the original question is answerable, please start the subquestion with "Now we can answer the question: <original question>".

main question: Who was the president in 1980 of the country that has Azad Kashmir?
subuestion 1: Which country contains Azad Kashmir?
subanswer 1: The answer is: Pakistan.
subquestion 2: Who was the president of Pakistan in 1980?
subanswer 2: The answer is: Muhammad Zia-ul-Haq.
subquestion 3: Now we can answer the question: Who was the president in 1980 of the country that has Azad Kashmir?
subanswer 3: The answer is: Muhammad Zia-ul-Haq."""
        existing_subquestions_and_subanswers, next_subquestion_id = concat_subqs_and_subas(question_list, answer_list)        
        io_input = (
            decompose_prompt
            + "\n\n"
            + f"Main question: {user_question}"
            + "\n"
            + existing_subquestions_and_subanswers
            + f"subquestion {next_subquestion_id}:"
        )
        io_output_list, stop_reason = self.io.generate(
            io_input,
            max_tokens=128,
            num_return=num_subq,
            stop_tokens=[
                'subanswer',
                '\n\n'
            ],
        )
        return io_output_list, stop_reason
        
        
    def suba(self, user_question: str, subquestion_list, subanswer_list, num_suba):
        decompose_prompt = """Given a question, please decompose it into sub-questions. For each sub-question, please answer it in one complete sentence.When the original question is answerable, please start the subquestion with "Now we can answer the question: <original question>".

main question: Who was the president in 1980 of the country that has Azad Kashmir?
subuestion 1: Which country contains Azad Kashmir?
subanswer 1: The answer is: Pakistan.
subquestion 2: Who was the president of Pakistan in 1980?
subanswer 2: The answer is: Muhammad Zia-ul-Haq.
subquestion 3: Now we can answer the question: Who was the president in 1980 of the country that has Azad Kashmir?
subanswer 3: The answer is: Muhammad Zia-ul-Haq. """

        existing_subquestions_and_subanswers, next_subquestion_id = concat_subqs_and_subas(subquestion_list, subanswer_list)
        io_input = (
            decompose_prompt
            + "\n\n"
            + f"Main question: {user_question}"
            + "\n"
            + existing_subquestions_and_subanswers
            #+ f"Question {self.question_index}.{next_subquestion_id}: "
            #+ subquestion
            + "\n"
            + f"Please use one complete sentence to answer the question.\nsubanswer {next_subquestion_id}:"
        )       
        output_list, stop_reason = self.io.generate(
            io_input,
            max_tokens=512,
            num_return=num_suba,
            #stop_tokens = 7841 #question
            # stop_tokens = 271 
            stop_tokens = ['subquestion','\n\n']
        
        )        
        
        return output_list, stop_reason
        
    def generate_subquestions(
        self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
    ):
        subquestion_list, subanswer_list, value_list = [], [], []

        # decompose_prompt = self.decompose_prompt if not paraphrased else self.decompose_prompt_rephrased
        decompose_prompt = '''Given a question, please decompose it into sub-questions.'''
    
        io_input = (
            decompose_prompt
            #'Decompose the question into sub-questions and **output only** a JSON array of strings, without any extra text.'  
            + "\n"
            + user_question
        )
        while True:
          io_output_list = self.io.generate(
              io_input,
              max_tokens=512,
              num_return=1,
              stop_tokens=[
                  "Answer",
                  "\n",
                  "The answer",
                  # f"Answer {self.question_index}.{next_subquestion_id}",
                  # f"Answer {self.question_index}.{next_subquestion_id}:",
                  # f"Answer {self.question_index}.{next_subquestion_id}: ",
              ],
          )
          if '?' in io_output_list[0]:
            break
            
        #print(f'io_output_list:{io_output_list}')
        io_output_list=io_output_list[0].replace('\n', '')
        subquestion_list = []
        #io_cleaned_output_list = re.findall(r'\d+\.\s*(.+)', io_output_list)
        sentences = re.split(r'(?<=[.?!])\s+', io_output_list)
        subquestion_list =[]
        for sentence in sentences:
          if '?' in sentence:
            subquestion_list.append(sentence.split('?')[0].strip())
        
        #questions = [s.strip() for s in sentences if s.strip().endswith('?')]
        if len(subquestion_list) == 0:
          print('re 모듈로 추출안됨')
        #sections = re.split(r'(?=\d+\.)', one_line)    
        
        #if len(subquestion_list) < 1:
            #subquestion_list = list(set([o.strip() for o in questions]))
        print(f"subquestion list: {subquestion_list}")
        return subquestion_list

    def subanswer(self,
        user_question: str,
       
        subquestion_list,
        num_subanswer):
        #! generate subanswers to the subquestions generated above
        io_input_list = []
        subanswer_list = []
        value_list = []
        for subquestion in subquestion_list:
          
            io_input = (  
                subquestion
                + "\n"
                + f"Please use one complete sentence to answer the question.")
            io_input_list.append(io_input)
 

        io_output_list = self.io.generate(
            io_input_list,
            max_tokens=512,
            num_return=num_subanswer,
            stop_tokens=['\n\n\n'
                # f"Question {self.question_index}.{next_subquestion_id + 1}",
            ]
        )
        cleaned_io_output_list = [
            [io_output.strip() for io_output in io_output_group] for io_output_group in io_output_list
        ]
        print(f"subquestion에 대한 raw 답변:{io_output_list}")

        for i, cleaned_io_output_group in enumerate(cleaned_io_output_list):
            try:
                most_likely_answer, likelihood = self._get_most_likely_answer(cleaned_io_output_group)
            except Exception as e:
                raise GeneratorError(
                    source="generate answer to subquestions",
                    io_input=io_input_list[i],
                    io_output_list=cleaned_io_output_group,
                )
            subanswer_list.append(most_likely_answer)
            value_list.append(likelihood)


        assert len(subquestion_list) == len(subanswer_list) == len(value_list)



        return cleaned_io_output_list,subquestion_list, subanswer_list, value_list
            
    def score(self,eval_prompt: str):
        
        output = llm_proposal_score(
            self.io.model,
            self.io.tokenizer,
            eval_prompt,
            )
        print(output)
        only_score_prompt = output + "\nTherefore, the final score"

        score = self.io.generate(
                  only_score_prompt,
                  num_return=1,
                  max_tokens=10,
                  stop_tokens=['\n','\n\n'],
                  
                  # stop_tokens='\n\n\n',
              )
        print(score)
        number = re.search(r'Score:\s*(\d+/\d+)', score[0])
        if number:
          number = number.group(1)
          #분수일시
          num_str, den_str = number.split('/')   # ["3", "5"]
          numerator   = float(num_str)              # 3.0
          denominator = float(den_str)              # 5.0
          
          score = numerator / denominator 
          #number = re.findall(r'\d+', score)        
          print(f'number:{number}')
  
          # if "Score:" in output:
          #     output.split
          return score
        return 'not scored'


    def final_output(self, user_question, subquestion_list,subanswer_list,context ='', get_most_likely=True):
              #! generate potential answer to the user question
        final_answer = []
        # if self.enable_potential_score:
        print('-----------------------최종 프롬프트-----------------------------')
        if context =='':
          potential_score_input = ""
        else:
          potential_score_input = "context: " + context + "\n"
        for subq, suba in zip(subquestion_list, subanswer_list):
            # if reach_terminal_subquestion(subq, user_question):
            #     final_answer.append(None)
            potential_score_input += "subquestion: " + subq + "\n" +"answer: " + suba + '\n'
        potential_score_input += "original question: " + user_question + "\n"
        potential_score_input += "Please answer the original question based on subquestion-answer pairs and context"
        print(f'최종 프름프트:{potential_score_input}')
        potential_score_output = self.io.generate(
            potential_score_input,
            num_return=3,
            max_tokens=512,
            #stop_tokens=self.fewshot_cot_config["stop_tokens"],
            stop_tokens='\n\n\n',
        )
        print('-----------------------final_output 답변생성 3개씩-----------------------------')
        print(f'output 3개:{potential_score_output}')
        potential_score_input2 = [
            "Question: "
            + user_question
            + "\nAnswer: "
            + z
            + '\n\n'
            #+ "\nTherefore, the final answer is"
            + 'Extract only the final answer, without restating the question or explanation. Provide the answer as a short, direct response.'
            for z in potential_score_output
        ]
        if get_most_likely:
            io_output_list = self.io.generate(
                potential_score_input2,
                num_return=1,
                max_tokens=40,
                stop_tokens='\n\n\n'
            )
            print('-----------------------final_output 답변요약-----------------------------')             
            print(f'get_most_likely 하기전의 final output(요약): {io_output_list}')
            cleaned_io_output_list = [
                [io_output.strip() for io_output in io_output_group] for io_output_group in io_output_list
            ]

            for i, cleaned_io_output_group in enumerate(cleaned_io_output_list):
                try:
                    most_likely_answer, likelihood = self._get_most_likely_answer(cleaned_io_output_group)
                except Exception as e:
                    raise GeneratorError(
                        source="generate answer to subquestions",
                        io_input=potential_score_input,
                        io_output_list=cleaned_io_output_group,
                    )
                final_answer.append(most_likely_answer)

        # final_answer.append(
        #     self.evaluator.extract_answer_from_model_completion(cleaned_io_output) 
        # )
        return potential_score_output, final_answer[0]
  
            



    def rephrased_question(self, user_question: str):
        rephrase_prompt ="""You are an AI assistant to help me rephrase questions by splitting the question context into conditions. In your rephrased question, remember to fully express the information in the original question.

        Examples:
        Original Question: Who was the president in 1980 of the country that has Azad Kashmir?
        Rephrased Question: Given a list of conditions, please answer the question. Condition 1: There is a country that has Azad Kashmir. Condition 2: We need to identify who was the president of that country in 1980. Question: Who was the president in 1980 of the country that has Azad Kashmir?

        Original Question: What is the mascot of the team that has Nicholas S. Zeppos as its leader?
        Rephrased Question: Given a list of conditions, please answer the question. Condition 1: Nicholas S. Zeppos serves as the leader of a particular team. Condition 2: We need to identify the mascot of that team. Question: What is the mascot of the team led by Nicholas S. Zeppos?"""

        rephrased_user_question_list = []
        io_input = rephrase_prompt

        io_input += "\n\n"
        io_input += "Rephrase Original Question: " + user_question + "\n"
        io_input += "Rephrased question you generate should start with Given a list of conditions, please answer the question. Condition 1:, and it should be one line"
        rephrased_question = self.io.generate(model_input=io_input, max_tokens=512, num_return=1, stop_tokens=[])[0]
        # io_output = self.io.generate(model_input=io_input, max_tokens=512, num_return=1, stop_tokens=[])[0]
        # io_output = "Given a list of conditions, please answer the question: " + user_question + " Condition 1:" + io_output.split("Condition 1:")[-1] if "Condition 1:" in io_output else "Given a list of conditions, please answer the question. Condition 1: " + io_output
        # rephrased_user_question_list.append(io_output)

        print(f"Rephrased user question is: {rephrased_user_question_list}")

    
        

        return rephrased_question

