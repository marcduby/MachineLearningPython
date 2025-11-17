
import networkx as nx
import matplotlib.pyplot as plt

# Create graph
G = nx.Graph()

# Define groups and genes
groups = {
    "Metal homeostasis": ['MT2A', 'SLC39A4', 'CP', 'CCS', 'S100A2', 'S100A7', 'S100A8', 'S100A9', 'S100A10'],
    "Zinc-dependent enzymes": ['CA1', 'CA2', 'CA3', 'CA14', 'HDAC1', 'HDAC4', 'HDAC8', 'PARP1'],
    "Oxidative stress": ['SOD1', 'TP53', 'PRDX1'],
    "Extracellular matrix": ['MMP9', 'FN1', 'APP'],
    "Immune complement": ['C1QA', 'C1QB', 'C1QC', 'C4A', 'C4B', 'C4BPA', 'C4BPB', 'C5', 'C8A', 'C8B', 'C8G'],
    "Coagulation factors": ['FGA', 'F12', 'F13B', 'KNG1', 'KLKB1'],
    "Lipid metabolism": ['APOE', 'APOA1', 'APOA2'],
    "Signal transduction": ['TP53', 'MDM2', 'ESR1', 'HDAC1', 'HDAC4', 'HDAC8'],
    "Neurotransmission": ['GLRA1']
}

# Add nodes and edges
for group, genes in groups.items():
    G.add_node(group, type='group')
    for gene in genes:
        G.add_node(gene, type='gene')
        G.add_edge(group, gene)

# Drawing
pos = nx.spring_layout(G, k=1.5)
plt.figure(figsize=(15, 12))

# Draw groups
group_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'group']
gene_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'gene']

nx.draw_networkx_nodes(G, pos, nodelist=group_nodes, node_size=2000, node_color='lightblue')
nx.draw_networkx_nodes(G, pos, nodelist=gene_nodes, node_size=500, node_color='lightgreen')
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos, font_size=10)

plt.title("Gene Network for Wilson's Disease Drug Targets", fontsize=16)
plt.axis('off')
plt.show()

