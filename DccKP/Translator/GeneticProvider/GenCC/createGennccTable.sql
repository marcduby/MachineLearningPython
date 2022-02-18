
drop table if exists data_gencc_gene_phenotype;
create table data_gencc_gene_phenotype (
  id                        int not null auto_increment primary key,
  excel_id                  varchar(1000) not null,                         -- uuid
  gene                      varchar(100) not null,                          -- gene_symbol
  gene_ncbi_id              varchar(100),                             -- TBD
  gene_hgnc_id              varchar(100),                                   -- gene_curie
  gene_annotation           varchar(100),                                   -- moi_title
  phenotype                 varchar(1000),                                  -- disease_title
  phenotype_mondo_id        varchar(100),                                   -- disease_curie
  phenotype_genepro_name    varchar(1000),                            -- FROM NODE NORMALIZER
  submitter                 varchar(100),                                   -- submitter_title
  submitter_curie           varchar(100),                                   -- submitter_curie
  score_classification      varchar(100),                                   -- classification_title
  gene_genepro_id           int(9),
  phenotype_genepro_id      int(9),
  score_genepro             double                                    -- classification calculated
);

-- add in study_secondary_id
alter table data_gencc_gene_phenotype add study_secondary_id int(3);
alter table data_gencc_gene_phenotype add publications varchar(1000);


-- update the secondary study id
update data_gencc_gene_phenotype dg join tran_test_202108.comb_study_type st 
on (dg.submitter COLLATE utf8mb4_unicode_ci = st.study_name)
set dg.study_secondary_id = st.study_id;

-- update the gene id
update data_gencc_gene_phenotype dg join tran_test_202108.comb_node_ontology node 
on (dg.gene_ncbi_id COLLATE utf8mb4_unicode_ci = node.ontology_id)
set dg.gene_genepro_id = node.id;

-- insert new genes that come from genCC
insert into tran_test_202108.comb_node_ontology (node_code, node_type_id, ontology_id, ontology_type_id, node_name, added_by_study_id)
select distinct gene, 2, gene_ncbi_id, 1, gene, 7 from data_gencc_gene_phenotype 
where gene_ncbi_id COLLATE utf8mb4_unicode_ci not in (select ontology_id from tran_test_202108.comb_node_ontology where node_type_id = 2);

-- update the disease genepro id
update data_gencc_gene_phenotype dg join tran_test_202108.comb_node_ontology node 
on (dg.phenotype_mondo_id COLLATE utf8mb4_unicode_ci = node.ontology_id)
set dg.phenotype_genepro_id = node.id;

-- insert new diseases that come from genCC
insert into tran_test_202108.comb_node_ontology (node_code, node_type_id, ontology_id, ontology_type_id, node_name, added_by_study_id)
select distinct phenotype_genepro_name, 1, phenotype_mondo_id, 2, phenotype_genepro_name, 7 from data_gencc_gene_phenotype 
where phenotype_mondo_id COLLATE utf8mb4_unicode_ci not in (select ontology_id from tran_test_202108.comb_node_ontology where node_type_id in (1, 3));

-- add translator score
      -- - add score: Definitive: 0.99 Strong: 0.8 Moderate: 0.5 Supportive: 0.25 limited: 0.10 Disputed: 0.05 Refuted: 0 Animal: 0.2
-- +-----------+-------------------------------+
-- |      1724 | Definitive                    |
-- |      1039 | Moderate                      |
-- |       745 | Limited                       |
-- |      2403 | Strong                        |
-- |       187 | No Known Disease Relationship |
-- |        15 | Disputed Evidence             |
-- |       264 | Supportive                    |
-- |         1 | Refuted Evidence              |
-- +-----------+-------------------------------+
update data_gencc_gene_phenotype set score_genepro = 0.99 where score_classification = 'Definitive';
update data_gencc_gene_phenotype set score_genepro = 0.8 where score_classification = 'Strong';
update data_gencc_gene_phenotype set score_genepro = 0.5 where score_classification = 'Moderate';
update data_gencc_gene_phenotype set score_genepro = 0.25 where score_classification = 'Supportive';
update data_gencc_gene_phenotype set score_genepro = 0.1 where score_classification = 'Limited';
update data_gencc_gene_phenotype set score_genepro = 0.05 where score_classification = 'Disputed Evidence';
update data_gencc_gene_phenotype set score_genepro = 0.0 where score_classification = 'Refuted Evidence';
update data_gencc_gene_phenotype set score_genepro = 0.0 where score_classification = 'No Known Disease Relationship';




-- insert into translator tables
-- +--------------------+---------------+------+-----+-------------------+-------------------+
-- | Field              | Type          | Null | Key | Default           | Extra             |
-- +--------------------+---------------+------+-----+-------------------+-------------------+
-- | id                 | int           | NO   | PRI | NULL              | auto_increment    |
-- | edge_id            | varchar(100)  | NO   |     | NULL              |                   |
-- | source_node_id     | int           | NO   | MUL | NULL              |                   |
-- | target_node_id     | int           | NO   | MUL | NULL              |                   |
-- | edge_type_id       | int           | NO   |     | NULL              |                   |
-- | score              | double        | YES  | MUL | NULL              |                   |
-- | score_text         | varchar(50)   | YES  |     | NULL              |                   |
-- | score_type_id      | int           | NO   | MUL | NULL              |                   |
-- | study_id           | int           | NO   |     | NULL              |                   |
-- | date_created       | datetime      | YES  |     | CURRENT_TIMESTAMP | DEFAULT_GENERATED |
-- | score_translator   | double        | YES  |     | NULL              |                   |
-- | study_secondary_id | int           | YES  |     | NULL              |                   |
-- | publication_ids    | varchar(1000) | YES  |     | NULL              |                   |
-- +--------------------+---------------+------+-----+-------------------+-------------------+

-- insert gene to disease
insert into tran_test_202108.comb_edge_node 
(edge_id, source_node_id, target_node_id, edge_type_id, score_text, score_type_id, study_id, study_secondary_id, score_translator, publication_ids)
select concat('gencc_ge_di_', excel_id), gene_genepro_id, phenotype_genepro_id, 5, score_classification, 11, 7, study_secondary_id, score_genepro, publications
from data_gencc_gene_phenotype;

-- insert disease to gene
insert into tran_test_202108.comb_edge_node 
(edge_id, source_node_id, target_node_id, edge_type_id, score_text, score_type_id, study_id, study_secondary_id, score_translator, publication_ids)
select concat('gencc_di_ge_', excel_id), phenotype_genepro_id, gene_genepro_id, 10, score_classification, 11, 7, study_secondary_id, score_genepro, publications
from data_gencc_gene_phenotype;


-- scratch
select count(*), study_id from tran_test_202108.comb_edge_node group by study_id;
delete from tran_test_202108.comb_edge_node where study_id = 7;


select gene, gene_hgnc_id, gene_ncbi_id, gene, phenotype_mondo_id, gene_genepro_id, phenotype_genepro_id, study_secondary_id
from data_gencc_gene_phenotype
limit 100;

-- find all gencc data that is missing gene in genepro table
select gene, gene_hgnc_id, gene_ncbi_id, gene, phenotype_mondo_id, gene_genepro_id, phenotype_genepro_id, study_secondary_id
from data_gencc_gene_phenotype
where gene_genepro_id is null
limit 100;

select distinct gene, gene_ncbi_id from data_gencc_gene_phenotype 
where gene_ncbi_id COLLATE utf8mb4_unicode_ci not in (select ontology_id from tran_test_202108.comb_node_ontology where node_type_id = 2);

-- find all gencc data that is disease in genepro table
select gene, gene_hgnc_id, gene_ncbi_id, gene, phenotype_mondo_id, gene_genepro_id, phenotype_genepro_id, study_secondary_id
from data_gencc_gene_phenotype
where phenotype_genepro_id is null
limit 1000;

select distinct phenotype_genepro_name, phenotype_mondo_id from data_gencc_gene_phenotype 
where phenotype_mondo_id COLLATE utf8mb4_unicode_ci not in (select ontology_id from tran_test_202108.comb_node_ontology where node_type_id in (1, 3));


select count(id), score_classification, score_genepro from data_gencc_gene_phenotype group by score_classification, score_genepro order by score_genepro desc;


select count(id), submitter, study_secondary_id from data_gencc_gene_phenotype group by submitter, study_secondary_id;

-- mysql> select count(id), submitter, study_secondary_id from data_gencc_gene_phenotype group by submitter, study_secondary_id;
-- +-----------+--------------------------------------------+--------------------+
-- | count(id) | submitter                                  | study_secondary_id |
-- +-----------+--------------------------------------------+--------------------+
-- |      1784 | Ambry Genetics                             |               NULL |
-- |      1238 | Genomics England PanelApp                  |               NULL |
-- |       162 | Illumina                                   |               NULL |
-- |       531 | Invitae                                    |               NULL |
-- |       148 | Myriad Women’s Health                      |               NULL |
-- |       639 | PanelApp Australia                         |               NULL |
-- |      1554 | TGMI|G2P                                   |               NULL |
-- |        58 | Franklin by Genoox                         |               NULL |
-- |       264 | Online Mendelian Inheritance in Man (OMIM) |               NULL |
-- +-----------+--------------------------------------------+--------------------+
-- 9 rows in set (0.02 sec)

-- +-----------+--------------------------------------------+--------------------+
-- | count(id) | submitter                                  | study_secondary_id |
-- +-----------+--------------------------------------------+--------------------+
-- |      1784 | Ambry Genetics                             |                  8 |
-- |      1238 | Genomics England PanelApp                  |                  9 |
-- |       162 | Illumina                                   |                 10 |
-- |       531 | Invitae                                    |                 11 |
-- |       148 | Myriad Women’s Health                      |                 12 |
-- |       639 | PanelApp Australia                         |                 13 |
-- |      1554 | TGMI|G2P                                   |                 14 |
-- |        58 | Franklin by Genoox                         |                 15 |
-- |       264 | Online Mendelian Inheritance in Man (OMIM) |                 16 |
-- +-----------+--------------------------------------------+--------------------+
-- 9 rows in set (0.02 sec)

