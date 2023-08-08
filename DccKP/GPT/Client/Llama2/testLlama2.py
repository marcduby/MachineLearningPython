
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
        temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        # max_new_tokens=512,  # mex number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating


    )
    print(f"Pipeline loaded on {device}")

    # test on simple question
    res = generate_text("Explain to me the difference between nuclear fission and fusion.")
    print(res[0]["generated_text"])


    # test on genetics summary
    input_text = '''
summarize the information related to MAP3K15 from the information below:
We performed collapsing analyses on 454,796 UK Biobank (UKB) exomes to detect gene-level associations with diabetes. Recessive carriers of nonsynonymous variants in  were 30% less likely to develop diabetes ( = 5.7 × 10) and had lower glycosylated hemoglobin (β = -0.14 SD units,  = 1.1 × 10). These associations were independent of body mass index, suggesting protection against insulin resistance even in the setting of obesity. We replicated these findings in 96,811 Admixed Americans in the Mexico City Prospective Study ( < 0.05)Moreover, the protective effect of  variants was stronger in individuals who did not carry the Latino-enriched  risk haplotype ( = 6.0 × 10). Separately, we identified a Finnish-enriched  protein-truncating variant associated with decreased odds of both type 1 and type 2 diabetes ( < 0.05) in FinnGen. No adverse phenotypes were associated with protein-truncating  variants in the UKB, supporting this gene as a therapeutic target for diabetes.
A major goal in human genetics is to use natural variation to understand the phenotypic consequences of altering each protein-coding gene in the genome. Here we used exome sequencing to explore protein-altering variants and their consequences in 454,787 participants in the UK Biobank study. We identified 12 million coding variants, including around 1 million loss-of-function and around 1.8 million deleterious missense variants. When these were tested for association with 3,994 health-related traits, we found 564 genes with trait associations at P ≤ 2.18 × 10. Rare variant associations were enriched in loci from genome-wide association studies (GWAS), but most (91%) were independent of common variant signals. We discovered several risk-increasing associations with traits related to liver disease, eye disease and cancer, among others, as well as risk-lowering associations for hypertension (SLC9A3R2), diabetes (MAP3K15, FAM234A) and asthma (SLC27A3). Six genes were associated with brain imaging phenotypes, including two involved in neural development (GBE1, PLD1). Of the signals available and powered for replication in an independent cohort, 81% were confirmed; furthermore, association signals were generally consistent across individuals of European, Asian and African ancestry. We illustrate the ability of exome sequencing to identify gene-trait associations, elucidate gene function and pinpoint effector genes that underlie GWAS signals at scale.
Mitogen-activated protein kinases (MAP kinases) are functionally connected kinases that regulate key cellular process involved in kidney disease such as all survival, death, differentiation and proliferation. The typical MAP kinase module is composed by a cascade of three kinases: a MAP kinase kinase kinase (MAP3K) that phosphorylates and activates a MAP kinase kinase (MAP2K) which phosphorylates a MAP kinase (MAPK). While the role of MAPKs such as ERK, p38 and JNK has been well characterized in experimental kidney injury, much less is known about the apical kinases in the cascade, the MAP3Ks. There are 24 characterized MAP3K (MAP3K1 to MAP3K21 plus RAF1, BRAF and ARAF). We now review current knowledge on the involvement of MAP3K in non-malignant kidney disease and the therapeutic tools available. There is in vivo interventional evidence clearly supporting a role for MAP3K5 (ASK1) and MAP3K14 (NIK) in the pathogenesis of experimental kidney disease. Indeed, the ASK1 inhibitor Selonsertib has undergone clinical trials for diabetic kidney disease. Additionally, although MAP3K7 (MEKK7, TAK1) is required for kidney development, acutely targeting MAP3K7 protected from acute and chronic kidney injury; and targeting MAP3K8 (TPL2/Cot) protected from acute kidney injury. By contrast MAP3K15 (ASK3) may protect from hypertension and BRAF inhibitors in clinical use may induced acute kidney injury and nephrotic syndrome. Given their role as upstream regulators of intracellular signaling, MAP3K are potential therapeutic targets in kidney injury, as demonstrated for some of them. However, the role of most MAP3K in kidney disease remains unexplored.
    '''

    res = generate_text(input_text)
    print("\n\n")
    print(res[0]["generated_text"])


    input_text = '''
summarize the following for gene UBE2NL from the information below:
Cancer is characterized by abnormal growth of cells. Targeting ubiquitin proteins in the discovery of new anticancer therapeutics is an attractive strategy. The present study uses the structure-based drug discovery methods to identify new lead structures, which are selective to the putative ubiquitin-conjugating enzyme E2N-like (UBE2NL). The 3D structure of the UBE2NL was evaluated using homology modeling techniques. The model was validated using standard in silico methods. The hydrophobic pocket of UBE2NL that aids in binding with its natural receptor ubiquitin-conjugating enzyme E2 variant (UBE2V) was identified through protein-protein docking study. The binding site region of the UBE2NL was identified using active site prediction tools. The binding site of UBE2NL which is responsible for cancer cell progression is considered for docking study. Virtual screening study with the small molecular structural database was carried out against the active site of UBE2NL. The ligand molecules that have shown affinity towards UBE2NL were considered for ADME prediction studies. The ligand molecules that obey the Lipinski's rule of five and Jorgensen's rule of three pharmacokinetic properties like human oral absorption etc. are prioritized. The resultant ligand molecules can be considered for the development of potent UBE2NL enzyme inhibitors for cancer therapy.Migraine without aura (MWO) is the most common among migraine group, and is mainly associated with genetic, physical and chemical factors, and hormonal changes. We aimed to identify novel non-synonymous mutations predisposing to the susceptibility to MWO in a Chinese sample using exome sequencing. Four patients with MWO from a family and four non-migraine subjects unrelated with these patients were genotyped using whole-exome sequencing. Bioinformatics analysis was used to screen possible susceptibility gene mutations, which were then verified by PCR. In four patients with MWO, six novel rare non-synonymous mutations were observed, including EDA2R (G170A), UBE2NL (T266G), GBP2 (A907G), EMR1 (C264G), CLCNKB (A1225G), and ARHGAP28 (C413G). It is worth stressing that GBP2 (A907G) was absent in any control subject. Multiple genes predispose to the susceptibility to MWO. ARHGAP28-, EMR1-, and GBP2-encoded proteins may affect angiokinesis, which supports vasogenic theory for the etiological hypothesis of this disease. CLCNKB-encoded protein may affect cell membrane potential, which is consistent with the cortical spreading depression theory. UBE2NL-encoded protein may regulate cellular responses to 5-hydroxytryptamine, which is in accordance with trigeminovascular reflex theory. EDA2R and UBE2NL are located on the X chromosome, which supports that this disease may have gender differences in genetic predisposition. Replication in larger sample size would significantly strengthen these findings.Sporadic Alzheimer disease (SAD) is the most prevalent neurodegenerative disorder. With the development of new generation DNA sequencing technologies, additional genetic risk factors have been described. Here we used various methods to process DNA sequencing data in order to gain further insight into this important disease. We have sequenced the exomes of brain samples from SAD patients and non-demented controls. Using either method, we found a higher number of single nucleotide variants (SNVs), from SAD patients, in genes present at the X chromosome. Using the most stringent method, we validated these variants by Sanger sequencing. Two of these gene variants, were found in loci related to the ubiquitin pathway (UBE2NL and ATXN3L), previously do not described as genetic risk factors for SAD.Genetic association studies for gastroschisis have highlighted several candidate variants. However, genetic basis in gastroschisis from noninvestigated heritable factors could provide new insights into the human biology for this birth defect. We aim to identify novel gastroschisis susceptibility variants by employing whole exome sequencing (WES) in a Mexican family with recurrence of gastroschisis.We employed WES in two affected half-sisters with gastroschisis, mother, and father of the proband. Additionally, functional bioinformatics analysis was based on SVS-PhoRank and Ensembl-Variant Effect Predictor. The latter assessed the potentially deleterious effects (high, moderate, low, or modifier impact) from exome variants based on SIFT, PolyPhen, dbNSFP, Condel, LoFtool, MaxEntScan, and BLOSUM62 algorithms. The analysis was based on the Human Genome annotation, GRCh37/hg19. Candidate genes were prioritized and manually curated based on significant phenotypic relevance (SVS-PhoRank) and functional properties (Ensembl-Variant Effect Predictor). Functional enrichment analysis was performed using ToppGene Suite, including a manual curation of significant Gene Ontology (GO) biological processes from functional similarity analysis of candidate genes.No single gene-disrupting variant was identified. Instead, 428 heterozygous variations were identified for which SPATA17, PDE4DIP, CFAP65, ALPP, ZNF717, OR4C3, MAP2K3, TLR8, and UBE2NL were predicted as high impact in both cases, mother, and father of the proband. PLOD1, COL6A3, FGFRL1, HHIP, SGCD, RAPGEF1, PKD1, ZFHX3, BCAS3, EVPL, CEACAM5, and KLK14 were segregated among both cases and mother. Multiple interacting background modifiers may regulate gastroschisis susceptibility. These candidate genes highlight a role for development of blood vessel, circulatory system, muscle structure, epithelium, and epidermis, regulation of cell junction assembly, biological/cell adhesion, detection/response to endogenous stimulus, regulation of cytokine biosynthetic process, response to growth factor, postreplication repair/protein K63-linked ubiquitination, protein-containing complex assembly, and regulation of transcription DNA-templated.Considering the likely gene-disrupting prediction results and similar biological pattern of mechanisms, we propose a joint "multifactorial model" in gastroschisis pathogenesis.
    '''
    res = generate_text(input_text)
    print("\n\n")
    print(res[0]["generated_text"])

    subtext = '''
Cancer is characterized by abnormal growth of cells. Targeting ubiquitin proteins in the discovery of new anticancer therapeutics is an attractive strategy. The present study uses the structure-based drug discovery methods to identify new lead structures, which are selective to the putative ubiquitin-conjugating enzyme E2N-like (UBE2NL). The 3D structure of the UBE2NL was evaluated using homology modeling techniques. The model was validated using standard in silico methods. The hydrophobic pocket of UBE2NL that aids in binding with its natural receptor ubiquitin-conjugating enzyme E2 variant (UBE2V) was identified through protein-protein docking study. The binding site region of the UBE2NL was identified using active site prediction tools. The binding site of UBE2NL which is responsible for cancer cell progression is considered for docking study. Virtual screening study with the small molecular structural database was carried out against the active site of UBE2NL. The ligand molecules that have shown affinity towards UBE2NL were considered for ADME prediction studies. The ligand molecules that obey the Lipinski's rule of five and Jorgensen's rule of three pharmacokinetic properties like human oral absorption etc. are prioritized. The resultant ligand molecules can be considered for the development of potent UBE2NL enzyme inhibitors for cancer therapy.Migraine without aura (MWO) is the most common among migraine group, and is mainly associated with genetic, physical and chemical factors, and hormonal changes. We aimed to identify novel non-synonymous mutations predisposing to the susceptibility to MWO in a Chinese sample using exome sequencing. Four patients with MWO from a family and four non-migraine subjects unrelated with these patients were genotyped using whole-exome sequencing. Bioinformatics analysis was used to screen possible susceptibility gene mutations, which were then verified by PCR. In four patients with MWO, six novel rare non-synonymous mutations were observed, including EDA2R (G170A), UBE2NL (T266G), GBP2 (A907G), EMR1 (C264G), CLCNKB (A1225G), and ARHGAP28 (C413G). It is worth stressing that GBP2 (A907G) was absent in any control subject. Multiple genes predispose to the susceptibility to MWO. ARHGAP28-, EMR1-, and GBP2-encoded proteins may affect angiokinesis, which supports vasogenic theory for the etiological hypothesis of this disease. CLCNKB-encoded protein may affect cell membrane potential, which is consistent with the cortical spreading depression theory. UBE2NL-encoded protein may regulate cellular responses to 5-hydroxytryptamine, which is in accordance with trigeminovascular reflex theory. EDA2R and UBE2NL are located on the X chromosome, which supports that this disease may have gender differences in genetic predisposition. Replication in larger sample size would significantly strengthen these findings.Sporadic Alzheimer disease (SAD) is the most prevalent neurodegenerative disorder. With the development of new generation DNA sequencing technologies, additional genetic risk factors have been described. Here we used various methods to process DNA sequencing data in order to gain further insight into this important disease. We have sequenced the exomes of brain samples from SAD patients and non-demented controls. Using either method, we found a higher number of single nucleotide variants (SNVs), from SAD patients, in genes present at the X chromosome. Using the most stringent method, we validated these variants by Sanger sequencing. Two of these gene variants, were found in loci related to the ubiquitin pathway (UBE2NL and ATXN3L), previously do not described as genetic risk factors for SAD.Genetic association studies for gastroschisis have highlighted several candidate variants. However, genetic basis in gastroschisis from noninvestigated heritable factors could provide new insights into the human biology for this birth defect. We aim to identify novel gastroschisis susceptibility variants by employing whole exome sequencing (WES) in a Mexican family with recurrence of gastroschisis.We employed WES in two affected half-sisters with gastroschisis, mother, and father of the proband. Additionally, functional bioinformatics analysis was based on SVS-PhoRank and Ensembl-Variant Effect Predictor. The latter assessed the potentially deleterious effects (high, moderate, low, or modifier impact) from exome variants based on SIFT, PolyPhen, dbNSFP, Condel, LoFtool, MaxEntScan, and BLOSUM62 algorithms. The analysis was based on the Human Genome annotation, GRCh37/hg19. Candidate genes were prioritized and manually curated based on significant phenotypic relevance (SVS-PhoRank) and functional properties (Ensembl-Variant Effect Predictor). Functional enrichment analysis was performed using ToppGene Suite, including a manual curation of significant Gene Ontology (GO) biological processes from functional similarity analysis of candidate genes.No single gene-disrupting variant was identified. Instead, 428 heterozygous variations were identified for which SPATA17, PDE4DIP, CFAP65, ALPP, ZNF717, OR4C3, MAP2K3, TLR8, and UBE2NL were predicted as high impact in both cases, mother, and father of the proband. PLOD1, COL6A3, FGFRL1, HHIP, SGCD, RAPGEF1, PKD1, ZFHX3, BCAS3, EVPL, CEACAM5, and KLK14 were segregated among both cases and mother. Multiple interacting background modifiers may regulate gastroschisis susceptibility. These candidate genes highlight a role for development of blood vessel, circulatory system, muscle structure, epithelium, and epidermis, regulation of cell junction assembly, biological/cell adhesion, detection/response to endogenous stimulus, regulation of cytokine biosynthetic process, response to growth factor, postreplication repair/protein K63-linked ubiquitination, protein-containing complex assembly, and regulation of transcription DNA-templated.Considering the likely gene-disrupting prediction results and similar biological pattern of mechanisms, we propose a joint "multifactorial model" in gastroschisis pathogenesis.    
'''
    
    input_text = '''
                Write a concise summary of the following text delimited by triple backquotes.
                ```{}```
    '''.format(subtext)
    res = generate_text(input_text)
    print("\n\n")
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




