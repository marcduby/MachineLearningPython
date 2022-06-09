

-- query to look through our gene/disease and pathway/disease links 
-- common disease
select count(*), nod.ontology_id, tar.node_type_id  
from comb_edge_node ed, comb_node_ontology nod, comb_node_ontology tar
where ed.source_node_id = nod.id 
and ed.target_node_id = tar.id
and nod.ontology_id in (
    'MONDO:0004975', 'HP:0011015', 'MONDO:0004975', 'MONDO:0005002', 'MONDO:0005155', 'HP:0003003', 'HP:0002014', 'HP:0002315',
    'MONDO:0002251', 'HP:0003124', 'HP:0000822', 'MONDO:0005812', 'HP:0001993', 'MONDO:0005377', 'MONDO:0008170', 'MONDO:0005147', 
    'MONDO:0005148', 'MONDO:0005260', 'HP:000071')
group by nod.ontology_id, tar.node_type_id;

-- rare diseases
select count(*), nod.ontology_id, tar.node_type_id  
from comb_edge_node ed, comb_node_ontology nod, comb_node_ontology tar
where ed.source_node_id = nod.id 
and ed.target_node_id = tar.id
and nod.ontology_id in ('MONDO:0008078', 'MONDO:0018997', 'MONDO:0008078', 'MONDO:0007035', 'MONDO:0011399', 'MONDO:0007739',
    'MONDO:0010298', 'MONDO:0019609', 'MONDO:0010526')
group by nod.ontology_id, tar.node_type_id;



