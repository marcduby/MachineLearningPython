

-- STEPS
-- 1 - add new phenotypes to comb_node_ontology tables
-- 1a - add field to clingen load table
-- 1b - find MONDO IDs that are in the db already
-- 1c - create code from description for phenotype if not already loaded
-- 1d - add disease to lookup table if not there yet
-- 2 - add the edge data

-- STEP 0 - load and remove all non mondo diseases
delete from clingen_gene_phenotype 
where SUBSTRING_INDEX(SUBSTRING_INDEX(phenotype_id, ':', 1), ':', -1) != 'MONDO'
or SUBSTRING_INDEX(SUBSTRING_INDEX(phenotype_id, ':', 1), ':', -1) is null;
select count(*) from clingen_gene_phenotype;

-- STEP 1a - alter the comb_node_edge table where the filter data is
alter table clingen_gene_phenotype add phenotype_code varchar(100);

-- STEP 1b - look at the filter data, find modo ids already used
update clingen_gene_phenotype cli inner join tran_test.comb_node_ontology ont
on cli.phenotype_id = ont.ontology_id
set cli.phenotype_code = ont.node_code;

-- STEP 1c - filter input ata to only add MONDO IDs since there seem to be duplicate rows for the other ontologies 
select count(gene_id) as gene_count, gene_id, phenotype_id, provenance, classification from clingen_gene_phenotype group by gene_id, phenotype_id, provenance, classification order by gene_count;
select SUBSTRING_INDEX(SUBSTRING_INDEX(phenotype_id, ':', 2), ':', -1) as mondo from clingen_gene_phenotype where SUBSTRING_INDEX(SUBSTRING_INDEX(phenotype_id, ':', 1), ':', -1) = 'MONDO' limit 10;

update clingen_gene_phenotype cli
set cli.phenotype_code = concat(provenance, '_', substring(replace(phenotype, ' ', '_'), 1, 20), '_', SUBSTRING_INDEX(SUBSTRING_INDEX(phenotype_id, ':', 2), ':', -1))
where cli.phenotype_code is null and SUBSTRING_INDEX(SUBSTRING_INDEX(phenotype_id, ':', 1), ':', -1) = 'MONDO';

select count(distinct phenotype_id) as gene_count, phenotype_code from clingen_gene_phenotype 
where SUBSTRING_INDEX(SUBSTRING_INDEX(phenotype_id, ':', 1), ':', -1) = 'MONDO'
group by phenotype_code order by gene_count;

-- STEP 1d - add new phenotypes in the production tables
insert into tran_test.comb_node_ontology (node_code, node_type_id, ontology_id, ontology_type_id, node_name)
select distinct phenotype_code, 1, phenotype_id, 2, phenotype from clingen_gene_phenotype
where phenotype_id not in (select ontology_id from tran_test.comb_node_ontology);

select count(distinct 1, phenotype_id, 2, phenotype) from clingen_gene_phenotype
where phenotype_id not in (select ontology_id from tran_test.comb_node_ontology);


-- STEP 2 - edge data
-- clingen
insert into tran_test.comb_node_edge (edge_id, source_code, target_code, edge_type_id, source_type_id, target_type_id, score_text, score_type_id, study_id)
select concat(provenance, '_', id), gene, phenotype_code,   -- edge_id, source_code, target_code
    5, 2, 1,                                                -- edge_type_id, source_type_id, target_type_id
    classification, 11, 5                                       -- score, score_type_id, study_id
from tran_dataload.clingen_gene_phenotype where provenance = 'Clingen'
and classification not in ('Disputed', 'Refuted', 'No Known Disease Relationship');


-- clinvar
insert into tran_test.comb_node_edge (edge_id, source_code, target_code, edge_type_id, source_type_id, target_type_id, score_text, score_type_id, study_id)
select concat(provenance, '_', id), gene, phenotype_code,   -- edge_id, source_code, target_code
    5, 2, 1,                                                -- edge_type_id, source_type_id, target_type_id
    classification, 11, 6                                       -- score, score_type_id, study_id
from tran_dataload.clingen_gene_phenotype where provenance = 'Clinvar';


-- add reverse data
insert into comb_node_edge (edge_id, source_code, target_code, edge_type_id, source_type_id, target_type_id, score, score_type_id, study_id)
select edge_id, target_code, source_code, if(edge_type_id = 5, 10, edge_type_id), target_type_id, source_type_id, score, score_type_id, study_id
from comb_node_edge where lower(edge_id) like 'clin%';
limit 3;

-- mysql> desc clingen_gene_phenotype;
-- +----------------+---------------+------+-----+---------+----------------+
-- | Field          | Type          | Null | Key | Default | Extra          |
-- +----------------+---------------+------+-----+---------+----------------+
-- | id             | int           | NO   | PRI | NULL    | auto_increment |
-- | gene           | varchar(100)  | NO   |     | NULL    |                |
-- | gene_id        | varchar(100)  | YES  |     | NULL    |                |
-- | phenotype      | varchar(1000) | YES  |     | NULL    |                |
-- | phenotype_id   | varchar(50)   | YES  |     | NULL    |                |
-- | provenance     | varchar(50)   | YES  |     | NULL    |                |
-- | classification | varchar(100)  | YES  |     | NULL    |                |
-- +----------------+---------------+------+-----+---------+----------------+
-- 7 rows in set (0.01 sec)


-- mysql> desc comb_node_edge;
-- +----------------+--------------+------+-----+---------+----------------+
-- | Field          | Type         | Null | Key | Default | Extra          |
-- +----------------+--------------+------+-----+---------+----------------+
-- | id             | int          | NO   | PRI | NULL    | auto_increment |
-- | edge_id        | varchar(100) | NO   |     | NULL    |                |
-- | source_code    | varchar(50)  | NO   | MUL | NULL    |                |
-- | target_code    | varchar(50)  | NO   | MUL | NULL    |                |
-- | edge_type_id   | int          | NO   | MUL | NULL    |                |
-- | source_type_id | int          | NO   | MUL | NULL    |                |
-- | target_type_id | int          | NO   | MUL | NULL    |                |
-- | score          | double       | YES  | MUL | NULL    |                |
-- | score_type_id  | int          | NO   | MUL | NULL    |                |
-- | study_id       | int          | NO   |     | NULL    |                |
-- +----------------+--------------+------+-----+---------+----------------+
-- 10 rows in set (0.00 sec)

-- mysql> 








