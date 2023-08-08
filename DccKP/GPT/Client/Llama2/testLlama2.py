
# imports
import torch
import transformers
from torch import cuda, bfloat16
import os
from transformers import AutoTokenizer

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
    # model_config = transformers.AutoConfig.from_pretrained(
    #     model_id,
    #     use_auth_token=hf_auth
    # )

    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     trust_remote_code=True,
    #     config=model_config,
    #     # quantization_config=bnb_config,
    #     device_map='auto',
    #     use_auth_token=hf_auth
    # )
    # model.eval()
    # print(f"Model loaded on {device}")

    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     model_id,
    #     use_auth_token=hf_auth
    # )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"Tokenizer loaded on {device}")


    generate_text = transformers.pipeline(
        model=model_id, 
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
        temperature=0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        # max_new_tokens=512,  # mex number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating


    )
    print(f"Pipeline loaded on {device}")

    # test on simple question
    res = generate_text("Explain to me the difference between nuclear fission and fusion.")
    print(res[0]["generated_text"])


    # test on genetics summary
    input_text = 
    '''
summarize the information related to MAP3K15 from the information below:
We performed collapsing analyses on 454,796 UK Biobank (UKB) exomes to detect gene-level associations with diabetes. Recessive carriers of nonsynonymous variants in  were 30% less likely to develop diabetes ( = 5.7 × 10) and had lower glycosylated hemoglobin (β = -0.14 SD units,  = 1.1 × 10). These associations were independent of body mass index, suggesting protection against insulin resistance even in the setting of obesity. We replicated these findings in 96,811 Admixed Americans in the Mexico City Prospective Study ( < 0.05)Moreover, the protective effect of  variants was stronger in individuals who did not carry the Latino-enriched  risk haplotype ( = 6.0 × 10). Separately, we identified a Finnish-enriched  protein-truncating variant associated with decreased odds of both type 1 and type 2 diabetes ( < 0.05) in FinnGen. No adverse phenotypes were associated with protein-truncating  variants in the UKB, supporting this gene as a therapeutic target for diabetes.
A major goal in human genetics is to use natural variation to understand the phenotypic consequences of altering each protein-coding gene in the genome. Here we used exome sequencing to explore protein-altering variants and their consequences in 454,787 participants in the UK Biobank study. We identified 12 million coding variants, including around 1 million loss-of-function and around 1.8 million deleterious missense variants. When these were tested for association with 3,994 health-related traits, we found 564 genes with trait associations at P ≤ 2.18 × 10. Rare variant associations were enriched in loci from genome-wide association studies (GWAS), but most (91%) were independent of common variant signals. We discovered several risk-increasing associations with traits related to liver disease, eye disease and cancer, among others, as well as risk-lowering associations for hypertension (SLC9A3R2), diabetes (MAP3K15, FAM234A) and asthma (SLC27A3). Six genes were associated with brain imaging phenotypes, including two involved in neural development (GBE1, PLD1). Of the signals available and powered for replication in an independent cohort, 81% were confirmed; furthermore, association signals were generally consistent across individuals of European, Asian and African ancestry. We illustrate the ability of exome sequencing to identify gene-trait associations, elucidate gene function and pinpoint effector genes that underlie GWAS signals at scale.
Mitogen-activated protein kinases (MAP kinases) are functionally connected kinases that regulate key cellular process involved in kidney disease such as all survival, death, differentiation and proliferation. The typical MAP kinase module is composed by a cascade of three kinases: a MAP kinase kinase kinase (MAP3K) that phosphorylates and activates a MAP kinase kinase (MAP2K) which phosphorylates a MAP kinase (MAPK). While the role of MAPKs such as ERK, p38 and JNK has been well characterized in experimental kidney injury, much less is known about the apical kinases in the cascade, the MAP3Ks. There are 24 characterized MAP3K (MAP3K1 to MAP3K21 plus RAF1, BRAF and ARAF). We now review current knowledge on the involvement of MAP3K in non-malignant kidney disease and the therapeutic tools available. There is in vivo interventional evidence clearly supporting a role for MAP3K5 (ASK1) and MAP3K14 (NIK) in the pathogenesis of experimental kidney disease. Indeed, the ASK1 inhibitor Selonsertib has undergone clinical trials for diabetic kidney disease. Additionally, although MAP3K7 (MEKK7, TAK1) is required for kidney development, acutely targeting MAP3K7 protected from acute and chronic kidney injury; and targeting MAP3K8 (TPL2/Cot) protected from acute kidney injury. By contrast MAP3K15 (ASK3) may protect from hypertension and BRAF inhibitors in clinical use may induced acute kidney injury and nephrotic syndrome. Given their role as upstream regulators of intracellular signaling, MAP3K are potential therapeutic targets in kidney injury, as demonstrated for some of them. However, the role of most MAP3K in kidney disease remains unexplored.
    '''



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




