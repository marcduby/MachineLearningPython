

from TCT import translator_metakg
from TCT import translator_kpinfo
import matplotlib.pyplot as plt
import networkx as nx



APInames, metaKG, Translator_KP_info= translator_metakg.load_translator_resources(use_new_metakg_url=True)

All_predicates = list(set(metaKG['Predicate']))
All_categories = list((set(list(set(metaKG['Subject']))+list(set(metaKG['Object'])))))
API_withMetaKG = list(set(metaKG['API']))

print("All predicates: {}".format(All_predicates))


#     # generate a dictionary of API and its predicates
# API_predicates = {}
# for api in API_withMetaKG:
#     API_predicates[api] = list(set(metaKG[metaKG['API'] == api]['Predicate']))

