from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM

def load_model_and_tokenizer():
    model_name = "Qwen/Qwen2.5-Math-PRM-7B"
    model = AutoModel.from_pretrained(model_name,
                                    device_map="auto",
                                    torch_dtype=torch.bfloat16,
                                    trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
    
def stop_reason(response_tokens):
  if tokenizer.eos_token_id in response_tokens:
      return "EOS"
  else:
      return "END OF STEP"
      
      
def LLM(model):
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
                  
if __name__ == '__main__':
  model, tokenizer = LLM('qwen2.5')
  output, ids = llm_proposal(model,tokenizer, 'what is your name?',stop_tokens=151645) 
  print(output)
  print(f'ids:{ids[0]}')
  stop = stop_reason(ids[0].tolist())
  print(stop)
  