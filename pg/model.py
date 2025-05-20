    
def llm_proposal(model=None,tokenizer=None,messages=None,temperature='0.8',model_name='qwen',n=1,max_new_tokens=512,stop_tokens = None):
    if model_name =='qwen':
        if isinstance(messages, str):
            prompt = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": messages}
            ]            
            output_list = []
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
            return output_list, generated_ids
        elif isinstance(messages, list):
            output_list = []
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
                for k in range(n):
                    generated_ids = model.generate(
                        **model_inputs, max_new_tokens=max_new_tokens,temperature=0.8)
                    
                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        
                    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    subanswer_list.append(response)
                output_list.append(subanswer_list)
            return output_list, generated_ids  