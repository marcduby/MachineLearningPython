





-- find all magma genes ranked for disease
select subject.ontology_id subject_id, subject.node_name as subject_name, target.ontology_id as target_id, 
    edge.score_translator, edge.score as pvalue, study.study_name
from comb_edge_node edge, comb_node_ontology subject, comb_node_ontology target, comb_study_type study
where edge.source_node_id = subject.id and edge.target_node_id = target.id
and target.ontology_id = 'MONDO:0004975'
and subject.node_type_id = 2
and edge.study_id = study.study_id
order by edge.score_translator desc, edge.score;


-- find all genes ranked for similar pathways for disease
-- NOTE: this will miss genes that are not in our node table
select pgene.ontology_id subject_id, subject.node_name as subject_name, target.ontology_id as target_id, 
    edge.score_translator, edge.score as pvalue, study.study_name
from comb_edge_node edge, comb_node_ontology subject, comb_node_ontology target, comb_study_type study,
  tran_upkeep.data_pathway_similarity sim, tran_upkeep.data_pathway_genes pathway_gene, tran_upkeep.data_pathway pathway,
  comb_node_ontology pgene
where edge.source_node_id = subject.id and edge.target_node_id = target.id
and target.ontology_id = 'MONDO:0004975'
and subject.node_type_id = 4
and subject.node_code COLLATE utf8mb4_general_ci = sim.subject_pathway_code and sim.obget_pathway_code = pathway.pathway_code and pathway.id = pathway_gene.pathway_id
    and pathway_gene.gene_code = pgene.ontology_id COLLATE utf8mb4_general_ci
and edge.study_id = study.study_id;

order by edge.score_translator desc, edge.score;


-- find all pathway similar for disease
select subject.ontology_id subject_id, subject.node_name as subject_name, target.ontology_id as target_id, 
    edge.score_translator, edge.score as pvalue, study.study_name
from comb_edge_node edge, comb_node_ontology subject, comb_node_ontology target, comb_study_type study
where edge.source_node_id = subject.id and edge.target_node_id = target.id
and target.ontology_id = 'MONDO:0004975'
and subject.node_type_id = 4
and edge.study_id = study.study_id
order by edge.score_translator desc, edge.score;


-- find all pathways through magma for disease 
select subject.ontology_id subject_id, subject.node_name as subject_name, target.ontology_id as target_id, 
    edge.score_translator, edge.score as pvalue, study.study_name
from comb_edge_node edge, comb_node_ontology subject, comb_node_ontology target, comb_study_type study
where edge.source_node_id = subject.id and edge.target_node_id = target.id
and target.ontology_id = 'MONDO:0004975'
and subject.node_type_id = 4
and edge.study_id = study.study_id
order by edge.score_translator desc, edge.score;



