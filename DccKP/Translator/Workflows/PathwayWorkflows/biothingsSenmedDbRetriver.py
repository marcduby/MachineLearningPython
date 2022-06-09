
# imports
import json
import sys 
import logging
import datetime 
import os
import requests 
from pathlib import Path 
import re
import csv
import pandas as pd

# constants
handler = logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)
dir_code = "/Users/mduby/Code/WorkspacePython/"
dir_code = "/home/javaprog/Code/PythonWorkspace/"
dir_data = "/Users/mduby//Data/Broad/"
dir_data = "/home/javaprog/Data/Broad/"
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/Translator/TranslatorLibraries')
import translator_libs as tl
location_servers = dir_code + "MachineLearningPython/DccKP/Translator/Misc/Json/trapiListServices.json"
date_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
location_results = dir_data + "Translator/Workflows/PathwayPpargT2d/SenmedDb/"
file_result = location_results + "allPapersSenmedDbUmls.csv"
url_biothings_senmeddb = "https://biothings.ncats.io/semmeddb/query?q=pmid:{}&size=100"
max_count = 200

# list of papers
map_papers = {}
map_papers['16150867'] = "3-phosphoinositide-dependent protein kinase-1 activates the peroxisome proliferator-activated receptor-gamma and promotes adipocyte differentiation, Yin "
map_papers['8001151'] = "Stimulation of adipogenesis in fibroblasts by PPAR gamma 2, a lipid-activated transcription factor, Tontonoz"
map_papers['12021175'] = "Gene expression profile of adipocyte differentiation and its regulation by peroxisome proliferator-activated receptor-gamma agonists, Gerhold"
map_papers['10339548'] = "A peroxisome proliferator-activated receptor gamma ligand inhibits adipocyte differentiation. Oberfield"
map_papers['7838715'] = "Adipocyte-specific transcription factor ARF6 is a heterodimeric complex of two nuclear hormone receptors, PPAR gamma and RXR alpha, Tontonoz"
map_papers['10622252'] = "Dominant negative mutations in human PPARgamma associated with severe insulin resistance, diabetes mellitus and hypertension, Barroso"
map_papers['9806549'] = "A Pro12Ala substitution in PPARgamma2 associated with decreased receptor activity, lower body mass index and improved insulin sensitivity, Deeb"
map_papers['25157153'] = "Rare variants in PPARG with decreased activity in adipocyte differentiation are associated with increased risk of type 2 diabetes, Majithia"


# 20220529 - new papers
map_papers['34900790'] = "The role of the PPARG (Pro12Ala) common genetic variant on type 2 diabetes mellitus risk"
map_papers['35462933'] = "PRDM16 Regulating Adipocyte Transformation and Thermogenesis: A Promising Therapeutic Target for Obesity and Diabetes"
map_papers['35364246'] = "Therapeutic implications of sonic hedgehog pathway in metabolic disorders: Novel target for effective treatment"
map_papers['35341481'] = "Loss of thymidine phosphorylase activity disrupts adipocyte differentiation and induces insulin-resistant lipoatrophic diabetes"
map_papers['35054888'] = "Effects of Isorhamnetin on Diabetes and Its Associated Complications: A Review of In Vitro and In Vivo Studies and a Post Hoc Transcriptome Analysis of Involved Molecular Pathways"
map_papers['34545810'] = "Impaired mRNA splicing and proteostasis in preadipocytes in obesity-related metabolic disease"
map_papers['33959308'] = "Curcumin improves adipocytes browning and mitochondrial function in 3T3-L1 cells and obese rodent model"
map_papers['14684744'] = "Dioxin increases C/EBPbeta transcription by activating cAMP/protein kinase A"
map_papers['14530861'] = "The FOXC2 -512C>T variant is associated with hypertriglyceridaemia and increased serum C-peptide in Danish Caucasian glucose-tolerant subjects"
map_papers['12855691'] = "Overexpression of sterol regulatory element-binding protein-1a in mouse adipose tissue produces adipocyte hypertrophy, increased fatty acid secretion, and fatty liver"
map_papers['12677228'] = "The Role of PPARgamma Ligands as Regulators of the Immune Response"
map_papers['11928067'] = "Pro12Ala polymorphism in the peroxisome proliferator-activated receptor-gamma2 (PPARgamma2) is associated with higher levels of total cholesterol and LDL-cholesterol in male caucasian type 2 diabetes patients"
map_papers['27909015'] = "Diabetic human adipose tissue-derived mesenchymal stem cells fail to differentiate in functional adipocytes"
map_papers['27815534'] = "Biological roles of microRNAs in the control of insulin secretion and action"
map_papers['27657995'] = "Effects of Streptozotocin-Induced Diabetes on Proliferation and Differentiation Abilities of Mesenchymal Stem Cells Derived from Subcutaneous and Visceral Adipose Tissues"
map_papers['27493874'] = "Diabetic mice exhibited a peculiar alteration in body composition with exaggerated ectopic fat deposition after muscle injury due to anomalous cell differentiation"
map_papers['27445976'] = "Cooperation between HMGA1 and HIF-1 Contributes to Hypoxia-Induced VEGF and Visfatin Gene Expression in 3T3-L1 Adipocytes"


def query_biothings(paper_id, paper_name, log=False):
    ''' 
    find the journal if in the results
    '''
    # initialize
    pubmed_id = 'PMID:' + paper_id
    list_results = []
    is_found = False
    url_query = url_biothings_senmeddb.format(paper_id)

    # log
    if log:
        print("looking for pubmed id: {}".format(url_query))    

    # query the service
    response = requests.get(url_query)

    # try and catch exception
    try:
        json_output = response.json()
        # if log:
        #     print("got result: \n{}".format(json_output))
    except ValueError:
        print("GOT ERROR: skipping")

    # pick put the data
    map_result = {'pubmed_id': paper_id, 'title': paper_name[0:20], 'predicate': None, 'subject': None, 'subject_type': None, 'object': None, 'object_type': None}
    if json_output:
        if isinstance(json_output, dict):
            if json_output.get('hits'):
                for child in json_output.get('hits'):
                    is_found = True
                    map_result = child.get('predicate')
                    map_result = {'pubmed_id': paper_id, 'title': paper_name[0:20], 'predicate': child.get('predicate'), 
                        'subj_umls': child.get('subject').get('umls'),
                        'subject': child.get('subject').get('name'), 'subject_type': child.get('subject').get('semantic_type_name'),
                        'obj_umls': child.get('object').get('umls'),
                       'object': child.get('object').get('name'), 'object_type': child.get('object').get('semantic_type_name'),}
                    list_results.append(map_result)

    # add to list
    if not is_found:
        list_results.append(map_result)

    # return
    return list_results

if __name__ == "__main__":
    # initialize
    count = 0
    list_result = []

    # loop through the paper ids
    for key, value in map_papers.items():
        # test the max count
        if count < max_count:
            count += 1

            # get the biothings data for the paper
            list_temp = query_biothings(key, value, log=True)

            # add to the results
            list_result = list_result + list_temp

    # print the results
    print("\n=====results")
    for child in list_result:
        print(child)

    # create dataframe
    df_papers = pd.DataFrame(list_result)
    #temporaly display 999 rows
    with pd.option_context('display.max_rows', 999):
        print (df_papers)

    # write out the file
    df_papers.to_csv(file_result, sep='\t')
    print("wrote out the file to: {}".format(file_result))

