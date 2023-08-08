
# imports
import torch
import transformers
from torch import cuda, bfloat16
import os

# constants
model_id = 'meta-llama/Llama-2-7b-chat-hf'
HUG_KEY = os.environ.get('HUG_KEY')


# main
if __name__ == "__main__":
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    device_name = torch.cuda.get_device_name()
    print(f"Using device: {device} ({device_name})\n\n")

    # bnb_config = transformers.BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type='nf4',
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=bfloat16
    # )

    hf_auth = HUG_KEY
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        # quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=hf_auth
    )
    model.eval()
    print(f"Model loaded on {device}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )

    generate_text = transformers.pipeline(
        model=model, 
        tokenizer=tokenizer,

        # added in to fix CUDA out of memory
        torch_dtype=torch.bfloat16,

        trust_remote_code=True,
        device_map="auto",
        max_length=2500,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,

        return_full_text=True,  
        task='text-generation',
        temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=512,  # mex number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating


    )

    res = generate_text("Explain to me the difference between nuclear fission and fusion.")
    print(res[0]["generated_text"])





    # device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    # device_name = torch.cuda.get_device_name()
    # print(f"Using device: {device} ({device_name})\n\n")

    # bnb_config = transformers.BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type='nf4',
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=bfloat16
    # )

    # hf_auth = HUG_KEY
    # model_config = transformers.AutoConfig.from_pretrained(
    #     model_id,
    #     use_auth_token=hf_auth
    # )

    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     trust_remote_code=True,
    #     config=model_config,
    #     quantization_config=bnb_config,
    #     device_map='auto',
    #     use_auth_token=hf_auth
    # )
    # model.eval()
    # print(f"Model loaded on {device}")

    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     model_id,
    #     use_auth_token=hf_auth
    # )

    # generate_text = transformers.pipeline(
    #     model=model, tokenizer=tokenizer,
    #     return_full_text=True,  
    #     task='text-generation',
    #     temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    #     max_new_tokens=512,  # mex number of tokens to generate in the output
    #     repetition_penalty=1.1  # without this output begins repeating
    # )

    # res = generate_text("Explain to me the difference between nuclear fission and fusion.")
    # print(res[0]["generated_text"])




