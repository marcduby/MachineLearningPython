


select subj.ontology_id, obj.ontology_id, edge.score, substring(obj.node_name, 1, 50)
from comb_edge_node edge, comb_node_ontology subj, comb_node_ontology obj
where edge.source_node_id = subj.id and edge.target_node_id = obj.id 
and subj.ontology_id = 'MONDO:0005148'
and obj.node_type_id = 4
order by edge.score;


select subj.ontology_id, obj.ontology_id, edge.score, substring(obj.node_name, 1, 50)
from comb_edge_node edge, comb_node_ontology subj, comb_node_ontology obj
where edge.source_node_id = subj.id and edge.target_node_id = obj.id 
and subj.ontology_id = 'MONDO:0005148'
and obj.node_type_id = 4
order by edge.score limit 30;



