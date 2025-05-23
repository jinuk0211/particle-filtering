
from itertools import accumulate

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModel
)

import torch.nn.functional as F
import re

CANDIDATE_TOKENS = [648, 387]
STEP_TAG_ID = 12902


def batched_math_shepherd_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    inputs,
    batch_size
):
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


class PRM:
    def __init__(self, search_config, **model_kwargs):
        self.search_config = search_config
        if search_config.prm_path == "PRIME-RL/EurusPRM-Stage2":
            self.model, self.ref_model, self.tokenizer = self.load_model_and_tokenizer(**model_kwargs)
        else:
            self.model, self.tokenizer = self.load_model_and_tokenizer(**model_kwargs)

    def load_model_and_tokenizer(
        self, **model_kwargs
    ):
        raise NotImplementedError

    def score(
        self, questions, outputs
    ):
        raise NotImplementedError


class QWEN_PRM(PRM):
    def __init__(self, search_config, **model_kwargs):
        super().__init__(search_config, **model_kwargs)
        self.model, self.tokenizer = self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self):
        model_name = "Qwen/Qwen2.5-Math-PRM-7B"
        model = AutoModel.from_pretrained(model_name,
                                        device_map="auto",
                                        torch_dtype=torch.bfloat16,
                                        trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    
    def score(
        self, questions, outputs, outputs_is_single_step = True
    ):
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
        for question, answers in zip(questions, outputs):
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
    def __init__(self, search_config, **model_kwargs):
        # override original init, because we need to load two models and a tokenizer
        super().__init__(search_config, **model_kwargs)
        self.model, self.ref_model, self.tokenizer = self.load_model_and_tokenizer()


    def load_model_and_tokenizer(self):
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
        self, questions, outputs):
        '''
        Score a batch of questions and their step-by-step outputs using PRIME scoring.
        questions: list of questions
        outputs: list of lists of N responses, where N answers correspond to 1 question. 
        '''



        # TODO: implement PRIME scoring
        # implement based on the commented example code above, and also the MathShepherd code
        # Prepare inputs by combining questions and outputs
        all_scores = []
        for question, answers in zip(questions, outputs):
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
    def load_model_and_tokenizer(self):
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
        self, questions, outputs
    ):
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




def load_prm(config):
    if config.prm_path == "peiyi9979/math-shepherd-mistral-7b-prm":
        return MathShepherd(config)

    if config.prm_path == "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data":
        return RLHFFlow(config)
    if config.prm_path == "PRIME-RL/EurusPRM-Stage2":
        return PRIME(config)

    if config.prm_path == "Qwen/Qwen2.5-Math-PRM-7B":
        return QWEN_PRM(config)

    raise NotImplementedError(f"PRM {config.prm_path} not implemented")