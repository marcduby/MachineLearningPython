
# imports
import json

# build the dict
type_translation_input = {
        # predicates
        'biolink:affected_by': 'affected_by',
        'biolink:affects': 'affects',
        'biolink:treated_by': 'treated_by',
        'biolink:treats': 'treats',
        'biolink:related_to': 'related_to',
        'biolink:correlated_with': 'correlated_with',
        'biolink:has_metabolite': 'has_metabolite',
        'biolink:has_evidence': 'has_evidence',
        'biolink:interacts_with': 'interacts_with',

        # categories
        'biolink:Gene': 'Gene',
        'biolink:ChemicalSubstance': 'ChemicalSubstance',
        'biolink:Disease': 'Disease',
        'biolink:Drug': 'Drug',
        'biolink:Pathway': 'Pathway',
        'biolink:MolecularEntity': 'MolecularEntity',
        'biolink:Assay': 'Assay',
    }

# reverse the type translation map for output
type_translation_output = dict((value, key) for key, value in type_translation_input.items())

cpd_curie_translation_input = {
        # compounds (biolink: molepro)
        'PUBCHEM.COMPOUND': 'CID',
        'CHEMBL.COMPOUND': 'ChEMBL',
        'DRUGBANK': 'DrugBank',
        'KEGG': 'KEGG.COMPOUND',
    }

# reverse the curie translation map for output
cpd_curie_translation_output = dict((value, key) for key, value in cpd_curie_translation_input.items())

target_curie_translation_input = {
        # targets (biolink: molepro)
        'CHEMBL.TARGET': 'ChEMBL'
    }

# reverse the curie translation map for output
target_curie_translation_output = dict((value, key) for key, value in target_curie_translation_input.items())

assay_curie_translation_input = {
        # assays (biolink: molepro)
        'CHEMBL.ASSAY': 'ChEMBL',
    }

# reverse the curie translation map for output
assay_curie_translation_output = dict((value, key) for key, value in assay_curie_translation_input.items())

curie_translation_input = {
    'biolink:ChemicalSubstance':cpd_curie_translation_input,
    'biolink:MolecularEntity':target_curie_translation_input,
    'biolink:Assay':assay_curie_translation_input,
}

curie_translation_output = {
    'biolink:ChemicalSubstance':cpd_curie_translation_output,
    'biolink:MolecularEntity':target_curie_translation_output,
    'biolink:Assay':assay_curie_translation_output,
}

biolinkMap = {
    'type_translation_input': type_translation_input,
    'cpd_curie_translation_input': cpd_curie_translation_input,
    'target_curie_translation_input': target_curie_translation_input,
    'assay_curie_translation_input': assay_curie_translation_input,
    'curie_translation_input': curie_translation_input,
    'curie_translation_output': curie_translation_output
}

# write out to file
outFile = '/home/javaprog/Data/Broad/Translator/Molepro/biolinkTranslation.json'
with open(outFile, 'w') as json_file:
    json.dump(biolinkMap, json_file, indent=4, separators=(',', ': '))
    print("wrote out dict to file {}".format(outFile))
