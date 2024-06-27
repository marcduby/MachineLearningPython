
# imports
import json

def call_api_hgvs_notations(hgvs_notations, hg=38):
    import requests, sys
    server = "https://rest.ensembl.org"
    if hg == 37:
        server = "https://grch37.rest.ensembl.org"
    ext = "/vep/human/hgvs"
    headers={ "Content-Type" : "application/json", "Accept" : "application/json"}
    params = {"uniprot":'-', "refseq":'-', 'hgvs':'-'}
    data = {"hgvs_notations": hgvs_notations}
    r = requests.post(server+ext, headers=headers, json=data, params=params)
    
    if not r.ok:
        r.raise_for_status()
        sys.exit()
    
    return r.json()



if __name__ == "__main__":
    # hgvs_notations = ['17:g.7673537G>A']
    # # hgvs_notations = df.HGVSc.tolist()

    # decoded = call_api_hgvs_notations(hgvs_notations)
    # print(json.dumps(decoded, indent=2))

    # 6:39033595:G:A
    # get 'hgvsp'
    
    hgvs_notations = ['6:g.39033595G>A', '6:g.39034072G>A']
    hgvs_notations = ['6:g.39034072G>A']
    decoded = call_api_hgvs_notations(hgvs_notations, hg=37)
    print(json.dumps(decoded, indent=2))
