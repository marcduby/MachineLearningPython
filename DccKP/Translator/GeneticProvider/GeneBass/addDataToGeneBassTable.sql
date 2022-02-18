

-- create the filtered table
drop table if exists data_genebass_gene_phenotype_good_prob;
create table data_genebass_gene_phenotype_good_prob as select * from data_genebass_gene_phenotype where probability >= 0.15;
alter table data_genebass_gene_phenotype_good_prob add disease_type_id int(9);
alter table data_genebass_gene_phenotype_good_prob add ontology_type_id int(9);


-- add the phenotype names if present already in the translator db
-- select
select distinct node.node_name, node.ontology_id 
from tran_test_202108.comb_node_ontology node, data_genebass_gene_phenotype_good_prob genebass
where genebass.phenotype_genepro_name is null 
and node.ontology_id = genebass.phenotype_ontology_id COLLATE utf8mb4_unicode_ci;

-- update the phenotype names
update data_genebass_gene_phenotype_good_prob gb join tran_test_202108.comb_node_ontology node 
on (gb.phenotype_ontology_id COLLATE utf8mb4_unicode_ci = node.ontology_id and gb.phenotype_genepro_name is null)
set gb.phenotype_genepro_name = node.node_name;


-- add the gene ncbi ids if present already in the translator db
select distinct gene from data_genebass_gene_phenotype_good_prob where gene_ncbi_id is null;

-- select
select distinct node.node_name, node.ontology_id 
from tran_test_202108.comb_node_ontology node, data_genebass_gene_phenotype_good_prob genebass
where genebass.gene_ncbi_id is null 
and node.node_type_id = 2
and node.node_code = genebass.gene COLLATE utf8mb4_unicode_ci
limit 20;

-- update the gene ncbi ids
update data_genebass_gene_phenotype_good_prob gb join tran_test_202108.comb_node_ontology node 
on (gb.gene COLLATE utf8mb4_unicode_ci = node.node_code and gb.gene_ncbi_id is null and node.node_type_id = 2)
set gb.gene_ncbi_id = node.ontology_id;

-- see what genebass ontology_ids are still unresolved
select count(id), gb.phenotype_genebass, gb.phenotype_ontology_id, gb.phenotype_genepro_name
from data_genebass_gene_phenotype_good_prob gb
where gb.phenotype_genepro_name is null
group by gb.phenotype_ontology_id, gb.phenotype_genepro_name, gb.phenotype_genebass
order by gb.phenotype_genepro_name;
-- node normlizer error
-- +-----------+--------------------------------------------------+-----------------------+------------------------+
-- | count(id) | phenotype_genebass                               | phenotype_ontology_id | phenotype_genepro_name |
-- +-----------+--------------------------------------------------+-----------------------+------------------------+
-- |        59 | Cystatin C                                       | MESH:D055316          | NULL                   |
-- |        34 | Alanine aminotransferase                         | NCIT:C38503           | NULL                   |
-- |        33 | Aspartate aminotransferase                       | NCIT:C64467           | NULL                   |
-- |        86 | Body mass index (BMI)                            | NCIT:C16358           | NULL                   |
-- |        19 | Rheumatoid factor                                | MESH:D012217          | NULL                   |
-- |        20 | Weight change during worst episode of depression | NCIT:C3445            | NULL                   |
-- +-----------+--------------------------------------------------+-----------------------+------------------------+
-- 6 rows in set (0.02 sec)
update data_genebass_gene_phenotype_good_prob gb
set gb.phenotype_genepro_name = gb.phenotype_genebass
where gb.phenotype_genepro_name is null;


-- see what genebass genes are still unresolved
select count(id), gb.gene
from data_genebass_gene_phenotype_good_prob gb
where gb.gene_ncbi_id is null
group by gb.gene;

select count(id)
from data_genebass_gene_phenotype_good_prob gb
where gb.gene_ncbi_id is null;


-- updating the data not found
-- set disease type for ontologies found
update data_genebass_gene_phenotype_good_prob gb join tran_test_202108.comb_node_ontology node 
on (gb.phenotype_ontology_id COLLATE utf8mb4_unicode_ci = node.ontology_id and node.node_type_id in (1, 3))
set gb.disease_type_id = node.node_type_id;

select count(id), gb.phenotype_genebass, gb.phenotype_ontology_id, gb.phenotype_genepro_name, gb.disease_type_id
from data_genebass_gene_phenotype_good_prob gb
where gb.disease_type_id is null
group by gb.phenotype_ontology_id, gb.phenotype_genepro_name, gb.phenotype_genebass, gb.disease_type_id
order by gb.phenotype_genepro_name;

-- set all MONDO to disease
update data_genebass_gene_phenotype_good_prob gb set gb.disease_type_id = 1
where gb.disease_type_id is null and locate('ONDO', gb.phenotype_ontology_id) > 0;
-- update one offs
update data_genebass_gene_phenotype_good_prob gb set gb.disease_type_id = 1
where gb.phenotype_ontology_id in ('HP:0001681', 'HP:0011675', 'NCIT:C37967', 'HP:0005679', 'HP:0000113');
-- rest as phenotypes
update data_genebass_gene_phenotype_good_prob gb set gb.disease_type_id = 3
where gb.disease_type_id is null;

select gb.phenotype_ontology_id, locate('ONDO', gb.phenotype_ontology_id), gb.disease_type_id
from data_genebass_gene_phenotype_good_prob gb 
where gb.disease_type_id is null and locate('ONDO', gb.phenotype_ontology_id) > 0;

-- insert the new ontology ids into the production geneticspro tables
select distinct gb.phenotype_ontology_id, gb.disease_type_id
from data_genebass_gene_phenotype_good_prob gb 
where gb.phenotype_ontology_id COLLATE utf8mb4_unicode_ci not in (select distinct ontology_id from tran_test_202108.comb_node_ontology where node_type_id in (1, 3));

-- update ontology type
select SUBSTRING_INDEX(SUBSTRING_INDEX(gb.phenotype_ontology_id, ':', 1), ':', -1) as prefix
from data_genebass_gene_phenotype_good_prob gb 
limit 20;

select distinct gb.phenotype_ontology_id, ont.ontology_name
from data_genebass_gene_phenotype_good_prob gb, tran_test_202108.comb_ontology_type ont
where SUBSTRING_INDEX(SUBSTRING_INDEX(gb.phenotype_ontology_id, ':', 1), ':', -1) COLLATE utf8mb4_unicode_ci = ont.prefix;

update data_genebass_gene_phenotype_good_prob gb join tran_test_202108.comb_ontology_type ont 
on (SUBSTRING_INDEX(SUBSTRING_INDEX(gb.phenotype_ontology_id, ':', 1), ':', -1) COLLATE utf8mb4_unicode_ci = ont.prefix)
set gb.ontology_type_id = ont.ontology_id;

-- insert genebass data into the translator tables
-- mysql> desc comb_node_ontology;
-- +-------------------+---------------+------+-----+-------------------+-----------------------------------------------+
-- | Field             | Type          | Null | Key | Default           | Extra                                         |
-- +-------------------+---------------+------+-----+-------------------+-----------------------------------------------+
-- | id                | int           | NO   | PRI | NULL              | auto_increment                                |
-- | node_code         | varchar(500)  | NO   | MUL | NULL              |                                               |
-- | node_type_id      | int           | NO   | MUL | NULL              |                                               |
-- | ontology_id       | varchar(50)   | YES  | MUL | NULL              |                                               |
-- | ontology_type_id  | varchar(50)   | YES  |     | NULL              |                                               |
-- | node_name         | varchar(1000) | YES  |     | NULL              |                                               |
-- | last_updated      | datetime      | YES  |     | CURRENT_TIMESTAMP | DEFAULT_GENERATED on update CURRENT_TIMESTAMP |
-- | added_by_study_id | int           | YES  |     | NULL              |                                               |
-- +-------------------+---------------+------+-----+-------------------+-----------------------------------------------+
-- 8 rows in set (0.00 sec)

insert into tran_test_202108.comb_node_ontology
(node_code, node_type_id, ontology_id, node_name, added_by_study_id)
select distinct gb.phenotype_genepro_name, gb.disease_type_id, gb.phenotype_ontology_id, 
    gb.phenotype_genepro_name, 17
from data_genebass_gene_phenotype_good_prob gb 
where gb.phenotype_ontology_id COLLATE utf8mb4_unicode_ci not in (select distinct ontology_id from tran_test_202108.comb_node_ontology where node_type_id in (1, 3));

-- add ontology type for diseases

-- add new genes
insert into tran_test_202108.comb_node_ontology
(node_code, node_type_id, ontology_id, ontology_type_id, node_name, added_by_study_id)
select distinct gb.gene, 2, gene_ncbi_id, 1, gb.gene, 17
from data_genebass_gene_phenotype_good_prob gb 
where gb.gene_ncbi_id COLLATE utf8mb4_unicode_ci not in (select distinct ontology_id from tran_test_202108.comb_node_ontology where node_type_id = 2);


-- insert disease/gene rows
insert into tran_test_202108.comb_edge_node 
(edge_id, source_node_id, target_node_id, edge_type_id, score, score_type_id, study_id, score_translator)
select distinct concat('genebass_', gb.id), so.id, ta.id, 10, gb.probability, 9, 17, gb.probability
from data_genebass_gene_phenotype_good_prob gb, tran_test_202108.comb_node_ontology so, tran_test_202108.comb_node_ontology ta
where gb.gene_ncbi_id COLLATE utf8mb4_unicode_ci = ta.ontology_id
and gb.phenotype_ontology_id COLLATE utf8mb4_unicode_ci = so.ontology_id 
and ta.node_type_id = 2 and so.node_type_id in (1, 3);


-- insert gene/disease rows
insert into tran_test_202108.comb_edge_node 
(edge_id, source_node_id, target_node_id, edge_type_id, score, score_type_id, study_id, score_translator)
select distinct concat('genebass_', gb.id), ta.id, so.id, 5, gb.probability, 9, 17, gb.probability
from data_genebass_gene_phenotype_good_prob gb, tran_test_202108.comb_node_ontology so, tran_test_202108.comb_node_ontology ta
where gb.gene_ncbi_id COLLATE utf8mb4_unicode_ci = ta.ontology_id
and gb.phenotype_ontology_id COLLATE utf8mb4_unicode_ci = so.ontology_id 
and ta.node_type_id = 2 and so.node_type_id in (1, 3);


limit 20;

-- test with MONDO:0003781













-- query
select gene, phenotype_ontology_id, phenotype_genepro_name, gb.probability
from data_genebass_gene_phenotype_good_prob gb
where gb.phenotype_ontology_id = 'MONDO:0005148'
order by gb.probability desc;

select gene, phenotype_ontology_id, phenotype_genepro_name, gb.probability
from data_genebass_gene_phenotype_good_prob gb
where gb.phenotype_ontology_id = 'MONDO:0005148'
order by gb.gene;

select count(id), phenotype_ontology_id, phenotype_genepro_name 
from data_genebass_gene_phenotype_good_prob 
where phenotype_ontology_id like 'MONDO%' 
group by phenotype_ontology_id, phenotype_genepro_name
order by phenotype_genepro_name;
