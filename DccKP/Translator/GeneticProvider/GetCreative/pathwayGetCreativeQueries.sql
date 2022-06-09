

select distinct drug.drug_id, drug.drug_name, gene.gene_id, gene.gene_name, 
    gene.score_translator as score, gene.pvalue, gene.study_name, drug.predicate
from tran_creative.genes_filtered_pathway_disease_magma gene, tran_creative.drug_gene drug
where gene.gene_id COLLATE utf8mb4_general_ci = drug.gene_id
and gene.disease_id = 'MONDO:0004975'
order by gene.score_translator desc, gene.pvalue limit 30;


select distinct drug.drug_id, drug.drug_name, gene.gene_id, gene.gene_name, 
    gene.score_translator as score, gene.pvalue, gene.study_name, drug.predicate
from tran_creative.genes_filtered_pathway_disease_magma gene, tran_creative.drug_gene drug
where gene.gene_id COLLATE utf8mb4_general_ci = drug.gene_id
and gene.disease_id = 'MONDO:0005002'
order by gene.score_translator desc, gene.pvalue limit 30;


select distinct drug.drug_id, drug.drug_name, gene.gene_id, gene.gene_name, 
    gene.score_translator as score, gene.pvalue, gene.study_name, drug.predicate
from tran_creative.genes_filtered_pathway_disease_magma gene, tran_creative.drug_gene drug
where gene.gene_id COLLATE utf8mb4_general_ci = drug.gene_id
and gene.disease_id = 'MONDO:0005002' and gene.gene_name != 'SLC6A4'
order by gene.score_translator desc, gene.pvalue limit 30;


select distinct drug.drug_id, drug.drug_name, gene.gene_id, gene.gene_name, 
    gene.score_translator as score, gene.study_name, drug.predicate
from tran_creative.genes_filtered_pathway_disease_magma gene, tran_creative.drug_gene drug
where gene.gene_id COLLATE utf8mb4_general_ci = drug.gene_id
and gene.disease_id = 'MONDO:0005148'
order by gene.score_translator desc limit 30;

-- TEMP TABLE SELECTS




-- TEMP TABLE COUNT SELECT
select count(distinct gene_name) as magma_genes, disease_id from tran_creative.gene_disease_magma
group by disease_id;

select count(distinct gene_name) as expanded_genes, disease_id from tran_creative.genes_distinct_pathway_disease_magma
group by disease_id;

select count(distinct gene_name) as pathway_genes, disease_id from tran_creative.genes_filtered_pathway_disease_magma
group by disease_id;

select count(distinct path_old_id) as magma_pathways, disease_id from tran_creative.pathway_disease_magma 
group by disease_id;

select count(distinct path_new_id) as linked_pathways, disease_id from tran_creative.pathway_disease_magma
group by disease_id;




-- TEMP TABLES
-- all magma genes for a disease
drop table if exists tran_creative.gene_disease_magma;
create table tran_creative.gene_disease_magma as 
select subject.ontology_id gene_id, subject.node_name as gene_name, target.ontology_id as disease_id, 
    target.node_name as disease_name,
    edge.score_translator, edge.score as pvalue, study.study_name
from comb_edge_node edge, comb_node_ontology subject, comb_node_ontology target, comb_study_type study
where edge.source_node_id = subject.id and edge.target_node_id = target.id
and target.ontology_id in ('MONDO:0004975', 'MONDO:0005148', 'MONDO:0005002')
and subject.node_type_id = 2
and edge.study_id = study.study_id
order by edge.score_translator desc, edge.score;


-- all similar pathways for a disease
drop table if exists tran_creative.pathway_disease_magma;
create table tran_creative.pathway_disease_magma as 
select subject.ontology_id pathway_id, subject.node_name as pathway_name, target.ontology_id as disease_id, 
    target.node_name as disease_name,
    edge.score_translator, edge.score as pvalue, study.study_name, 
    sim.subject_pathway_code path_old_id, 
    sim.object_pathway_code path_new_id, data_path.pathway_updated_name path_new_name,
    sim.google_distance_min goog_dist_min
from comb_edge_node edge, comb_node_ontology subject, comb_node_ontology target, comb_study_type study,
    tran_upkeep.data_pathway_similarity sim, tran_upkeep.data_pathway data_path
where edge.source_node_id = subject.id and edge.target_node_id = target.id
and target.ontology_id in ('MONDO:0004975', 'MONDO:0005148', 'MONDO:0005002')
and subject.node_type_id = 4
and subject.node_code COLLATE utf8mb4_general_ci = sim.subject_pathway_code
    and sim.google_distance_min > 0.8 and sim.subject_pathway_code != sim.object_pathway_code
    and sim.object_pathway_code = data_path.pathway_code
and edge.study_id = study.study_id
order by edge.score_translator desc, edge.score;

alter table tran_creative.pathway_disease_magma add index creat_new_path_code_idx (path_new);


-- all genes from similar pathways from disease
drop table if exists tran_creative.genes_pathway_disease_magma;
create table tran_creative.genes_pathway_disease_magma as 
select new_path.*, gene.gene_code
from tran_upkeep.data_pathway_genes gene, tran_creative.pathway_disease_magma new_path, tran_upkeep.data_pathway path_data
where new_path.path_new_id = path_data.pathway_code and gene.pathway_id = path_data.id;

-- all DISTINCT genes from similar pathways from disease
drop table if exists tran_creative.genes_distinct_pathway_disease_magma;
create table tran_creative.genes_distinct_pathway_disease_magma as 
select distinct gene_code as gene_name, target_id as disease_id, target_name as disease_name
from tran_creative.genes_pathway_disease_magma;

alter table tran_creative.genes_distinct_pathway_disease_magma add index creat_new_gene_path_code_idx (gene_name);
alter table tran_creative.genes_distinct_pathway_disease_magma add index creat_new_dis_path_code_idx (disease_id);

-- filter distinct pathway expended genes by genetic data we have
drop table if exists tran_creative.genes_filtered_pathway_disease_magma;
create table tran_creative.genes_filtered_pathway_disease_magma as 
select magma_gene.*
from tran_creative.gene_disease_magma as magma_gene,
    tran_creative.genes_distinct_pathway_disease_magma pathway_gene
where magma_gene.gene_name COLLATE utf8mb4_general_ci = pathway_gene.gene_name
and magma_gene.disease_id COLLATE utf8mb4_general_ci = pathway_gene.disease_id
order by magma_gene.score_translator desc, magma_gene.pvalue;






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
-- 2 parts
select subject.ontology_id subject_id, substring(subject.node_name, 1, 10) as subject_name, target.ontology_id as target_id, 
    edge.score_translator, edge.score as pvalue, study.study_name, 
    sim.subject_pathway_code path_old, sim.object_pathway_code path_new, sim.google_distance_min dist
from comb_edge_node edge, comb_node_ontology subject, comb_node_ontology target, comb_study_type study,
    tran_upkeep.data_pathway_similarity sim
where edge.source_node_id = subject.id and edge.target_node_id = target.id
and target.ontology_id = 'MONDO:0004975'
and subject.node_type_id = 4
and subject.node_code COLLATE utf8mb4_general_ci = sim.subject_pathway_code
    and sim.google_distance_min > 0.8 and sim.subject_pathway_code != sim.object_pathway_code
and edge.study_id = study.study_id
order by edge.score_translator desc, edge.score;


-- SLOW
-- NOTE: this will miss genes that are not in our node table
select pgene.ontology_id subject_id, subject.node_name as subject_name, target.ontology_id as target_id, 
    edge.score_translator, edge.score as pvalue, study.study_name
from comb_edge_node edge, comb_node_ontology subject, comb_node_ontology target, comb_study_type study,
  tran_upkeep.data_pathway_similarity sim, tran_upkeep.data_pathway_genes pathway_gene, tran_upkeep.data_pathway pathway,
  comb_node_ontology pgene
where edge.source_node_id = subject.id and edge.target_node_id = target.id
and target.ontology_id = 'MONDO:0004975'
and subject.node_type_id = 4
and subject.node_code COLLATE utf8mb4_general_ci = sim.subject_pathway_code and sim.object_pathway_code = pathway.pathway_code and pathway.id = pathway_gene.pathway_id
    and pathway_gene.gene_code = pgene.ontology_id COLLATE utf8mb4_general_ci
    and sim.google_distance_min > 0.8 and sim.subject_pathway_code != sim.object_pathway_code
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



