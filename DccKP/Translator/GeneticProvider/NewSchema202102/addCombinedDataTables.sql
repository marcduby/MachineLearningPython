
-- DESIGN
-- for phenotypes/disease, needed a unique code for each; since not all ontologies cover all entries, but our DCC codes do
-- the join will be done on the DCC phenotype/disease code
-- same for gene

-- add a combined node-0edge-node tables
drop table if exists comb_node_edge;
create table comb_node_edge (
  id                        int not null auto_increment primary key,
  edge_id                   varchar(100) not null,
  source_code               varchar(50) not null,
  target_code               varchar(50) not null,
  edge_type_id              int(3) not null,
  source_type_id            int(3) not null,
  target_type_id            int(3) not null,
  score                     double,
  score_type_id             int(3) not null,
  study_id                  int(3) not null
);

ALTER TABLE comb_node_edge add date_created datetime DEFAULT CURRENT_TIMESTAMP;

-- add in table to link node codes (PPARG/BMI) to onltology ids returned to API queries
drop table if exists comb_node_ontology;
create table comb_node_ontology (
  id                        int not null auto_increment primary key,
  node_code                 varchar(50) not null,
  node_type_id              int(3) not null,
  ontology_id               varchar(50) not null,
  ontology_type_id          int(3) not null,
  node_name                 varchar(1000)
);
-- indices
alter table comb_node_ontology add index node_ont_node_cde_idx (node_code);
alter table comb_node_ontology add index node_ont_node_typ_idx (node_type_id);
alter table comb_node_ontology add index node_ont_ont_idx (ontology_id);
-- 20210519 - make ontology_id nullable for adding in new phenotypes
alter table comb_node_ontology modify ontology_id varchar(50) null;
alter table comb_node_ontology modify ontology_type_id varchar(50) null;

-- add node/edge type lookup tables
drop table if exists comb_lookup_type;
create table comb_lookup_type (
  type_id                   int not null primary key,
  type_name                 varchar(100) not null,
  type_family               enum('node', 'edge', 'attribute')
);

insert into comb_lookup_type values(1, 'biolink:Disease', 'node');
insert into comb_lookup_type values(2, 'biolink:Gene', 'node');
insert into comb_lookup_type values(3, 'biolink:PhenotypicFeature', 'node');
insert into comb_lookup_type values(4, 'biolink:Pathway', 'node');
insert into comb_lookup_type values(5, 'biolink:gene_associated_with_condition', 'edge');
insert into comb_lookup_type values(6, 'biolink:genetic_association', 'edge');
insert into comb_lookup_type values(7, 'biolink:symbol', 'attribute');
insert into comb_lookup_type values(8, 'biolink:p_value', 'attribute');
insert into comb_lookup_type values(9, 'biolink:probability', 'attribute');
insert into comb_lookup_type values(10, 'biolink:condition_associated_with_gene', 'edge');
-- 20210513 - add in score_text 
insert into comb_lookup_type values(11, 'biolink:classification', 'attribute');
-- 20210519 - added new row type
insert into comb_lookup_type values(12, 'biolink:DiseaseOrPhenotypicFeature', 'node');

-- insert into comb_lookup_type values(1, 'biolink:condition_associated_with_gene', 'edge');
-- insert into comb_lookup_type values(1, 'biolink:condition_associated_with_gene', 'edge');
-- insert into comb_lookup_type values(1, 'biolink:condition_associated_with_gene', 'edge');
-- insert into comb_lookup_type values(1, 'biolink:condition_associated_with_gene', 'edge');

-- add study lookup tables
drop table if exists comb_study_type;
create table comb_study_type (
  study_id                  int not null primary key,
  study_name                varchar(100) not null,
  publication               varchar(4000),
  description               varchar(4000)
);

insert into comb_study_type (study_id, study_name) values(1, 'Magma');
insert into comb_study_type (study_id, study_name) values(2, 'ABC');
insert into comb_study_type (study_id, study_name) values(3, 'Integrated Genetics');
insert into comb_study_type (study_id, study_name) values(4, 'Richards Effector Gene');
-- 20210513 - adding new provenances
insert into comb_study_type (study_id, study_name) values(5, 'ClinGen');
insert into comb_study_type (study_id, study_name) values(6, 'ClinVar');

-- add ontology lookup tables
drop table if exists comb_ontology_type;
create table comb_ontology_type (
  ontology_id               int not null primary key,
  ontology_name             varchar(100) not null,
  url                       varchar(4000),
  description               varchar(4000)
);

insert into comb_ontology_type (ontology_id, ontology_name) values(1, 'NCBI Gene');
insert into comb_ontology_type (ontology_id, ontology_name) values(2, 'MONDO disease/phenotype');
insert into comb_ontology_type (ontology_id, ontology_name) values(3, 'EFO disease/phenotype');
insert into comb_ontology_type (ontology_id, ontology_name) values(4, 'GO pathway');
insert into comb_ontology_type (ontology_id, ontology_name) values(5, 'UMLS disease/phenotype');
insert into comb_ontology_type (ontology_id, ontology_name) values(6, 'HP disease/phenotype');
insert into comb_ontology_type (ontology_id, ontology_name) values(7, 'NCIT disease/phenotype');









-- INSERTS ---
-- insert the genes
insert into comb_node_ontology (node_code, node_type_id, ontology_id, ontology_type_id, node_name)
select gene, 2, ncbi_id, 1, gene from gene_lookup;

-- insert the phenotype/diseases
-- mondo disease
insert into comb_node_ontology (node_code, node_type_id, ontology_id, ontology_type_id, node_name)
select phenotype_code, 1, mondo_id, 2, phenotype from phenotype_lookup where mondo_id is not null and category='Disease';
-- mondo phenotype
insert into comb_node_ontology (node_code, node_type_id, ontology_id, ontology_type_id, node_name)
select phenotype_code, 3, mondo_id, 2, phenotype from phenotype_lookup where mondo_id is not null and category in ('Phenotype', 'Measurement');
-- efo disease
insert into comb_node_ontology (node_code, node_type_id, ontology_id, ontology_type_id, node_name)
select phenotype_code, 1, efo_id, 3, phenotype from phenotype_lookup where efo_id is not null and category='Disease';
-- efo phenotype
insert into comb_node_ontology (node_code, node_type_id, ontology_id, ontology_type_id, node_name)
select phenotype_code, 3, efo_id, 3, phenotype from phenotype_lookup where efo_id is not null and category in ('Phenotype', 'Measurement');

-- fix '-' with ':' for onltology IDs
update comb_node_ontology set ontology_id = replace(ontology_id, '_', ':');

-- insert the pathways
-- GO pathways
insert into comb_node_ontology (node_code, node_type_id, ontology_id, ontology_type_id, node_name)
select distinct PATHWAY, 4, PATHWAY, 4, PATHWAY from MAGMA_PATHWAYS;


-- add a combined node-0edge-node tables
drop table if exists comb_node_edge;
create table comb_node_edge (
  id                        int not null auto_increment primary key,
  edge_id                   varchar(100) not null,
  source_code               varchar(50) not null,
  target_code               varchar(50) not null,
  edge_type_id              int(3) not null,
  source_type_id            int(3) not null,
  target_type_id            int(3) not null,
  score                     double,
  score_type_id             int(3) not null,
  study_id                  int(3) not null
);
-- add indices
alter table comb_node_edge add index comb_nod_edg_src_cde_idx (source_code);
alter table comb_node_edge add index comb_nod_edg_tgt_cde_idx (target_code);
alter table comb_node_edge add index comb_nod_edg_edg_typ_idx (edge_type_id);
alter table comb_node_edge add index comb_nod_edg_src_typ_idx (source_type_id);
alter table comb_node_edge add index comb_nod_edg_tgt_typ_idx (target_type_id);
alter table comb_node_edge add index comb_nod_edg_sco_idx (score);
alter table comb_node_edge add index comb_nod_edg_sco_typ_idx (score_type_id);

-- 20210513 - add in score_text for non numeric scores
alter table comb_node_edge add score_text varchar(50);

-- alter table comb_node_edge add foreign key (source_code) references comb_node_ontology(node_code);
-- alter table comb_node_edge add foreign key (target_code) references comb_node_ontology(node_code);
-- alter table comb_node_edge add foreign key (edge_type_id) references comb_lookup_type(type_id);
-- alter table comb_node_edge add foreign key (source_type_id) references comb_node_ontology(node_type_id);
-- alter table comb_node_edge add foreign key (target_type_id) references comb_node_ontology(node_type_id);


-- EDGE DATA
-- insert richards data - disease
insert into comb_node_edge (edge_id, source_code, target_code, edge_type_id, source_type_id, target_type_id, score, score_type_id, study_id)
select id, gene_name, phenotype_name, 5, 2, 1, probability, 9, 4
from richards_gene where category = 'disease';
-- insert richards data - phenotype
insert into comb_node_edge (edge_id, source_code, target_code, edge_type_id, source_type_id, target_type_id, score, score_type_id, study_id)
select id, gene_name, phenotype_name, 5, 2, 3, probability, 9, 4
from richards_gene where category = 'phenotypic_feature';
-- fix richards data phenotype codes
update comb_node_edge set target_code = 'Thyroid' where target_code = 'lowtsh' and edge_id like 'RC_GENES%';
update comb_node_edge set target_code = 'BILIRUBIN' where target_code = 'dbilirubin' and edge_id like 'RC_GENES%';
update comb_node_edge set target_code = 'LDL' where target_code = 'ldl' and edge_id like 'RC_GENES%';
update comb_node_edge set target_code = 'T2D' where target_code = 't2d' and edge_id like 'RC_GENES%';
update comb_node_edge set target_code = 'eBMD' where target_code = 'ebmd' and edge_id like 'RC_GENES%';
update comb_node_edge set target_code = 'DBP' where target_code = 'dbp' and edge_id like 'RC_GENES%';
update comb_node_edge set target_code = 'Ca' where target_code = 'calcium' and edge_id like 'RC_GENES%';
update comb_node_edge set target_code = 'RedCount' where target_code = 'rbc' and edge_id like 'RC_GENES%';
update comb_node_edge set target_code = 'SBP' where target_code = 'sbp' and edge_id like 'RC_GENES%';
update comb_node_edge set target_code = 'TG' where target_code = 'tg' and edge_id like 'RC_GENES%';
update comb_node_edge set target_code = 'HEIGHT' where target_code = 'height' and edge_id like 'RC_GENES%';
update comb_node_edge set target_code = 'FG' where target_code = 'glucose' and edge_id like 'RC_GENES%';

-- insert magma pathway data - disease
insert into comb_node_edge (edge_id, source_code, target_code, edge_type_id, source_type_id, target_type_id, score, score_type_id, study_id)
select pa.ID, pa.PATHWAY, co.node_code, 6, 4, 1, pa.PVALUE, 8, 1
from MAGMA_PATHWAYS pa, comb_node_ontology co where pa.DISEASE = co.ontology_id and co.node_type_id = 1 and CATEGORY = 'disease';
-- insert magma pathway data - phenotype
insert into comb_node_edge (edge_id, source_code, target_code, edge_type_id, source_type_id, target_type_id, score, score_type_id, study_id)
select pa.ID, pa.PATHWAY, co.node_code, 6, 4, 3, pa.PVALUE, 8, 1
from MAGMA_PATHWAYS pa, comb_node_ontology co where pa.DISEASE = co.ontology_id and co.node_type_id = 3 and CATEGORY = 'phenotypic_feature';

-- insert integrated genetics gene data - disease
insert into comb_node_edge (edge_id, source_code, target_code, edge_type_id, source_type_id, target_type_id, score, score_type_id, study_id)
select pa.ID, ge.node_code, co.node_code, 5, 2, 1, pa.SCORE, 9, 3
from SCORE_GENES pa, comb_node_ontology co, comb_node_ontology ge 
where pa.DISEASE = co.ontology_id and co.node_type_id = 1 and pa.GENE = ge.ontology_id and ge.node_type_id = 2 and CATEGORY = 'disease';
-- insert integrated genetics gene data - phenotype
insert into comb_node_edge (edge_id, source_code, target_code, edge_type_id, source_type_id, target_type_id, score, score_type_id, study_id)
select pa.ID, ge.node_code, co.node_code, 5, 2, 3, pa.SCORE, 9, 3
from SCORE_GENES pa, comb_node_ontology co, comb_node_ontology ge  
where pa.DISEASE = co.ontology_id and co.node_type_id = 3 and pa.GENE = ge.ontology_id and ge.node_type_id = 2 and CATEGORY = 'phenotypic_feature';

-- insert magma gene data - disease
insert into comb_node_edge (edge_id, source_code, target_code, edge_type_id, source_type_id, target_type_id, score, score_type_id, study_id)
select concat('magma_gene_',id), gene, phenotype_code, 5, 2, 1, p_value, 8, 1
from magma_gene_phenotype where biolink_category = 'biolink:Disease';
-- insert magma gene data - phenotype
insert into comb_node_edge (edge_id, source_code, target_code, edge_type_id, source_type_id, target_type_id, score, score_type_id, study_id)
select concat('magma_gene_',id), gene, phenotype_code, 5, 2, 3, p_value, 8, 1
from magma_gene_phenotype where biolink_category = 'biolink:PhenotypicFeature';

-- insert the reverse data
insert into comb_node_edge (edge_id, source_code, target_code, edge_type_id, source_type_id, target_type_id, score, score_type_id, study_id)
select edge_id, target_code, source_code, if(edge_type_id = 5, 10, edge_type_id), target_type_id, source_type_id, score, score_type_id, study_id
from comb_node_edge;
limit 3;

-- https://stackoverflow.com/questions/3164505/mysql-insert-record-if-not-exists-in-table




-- TODO
-- indexes
-- flipped query
-- DONE - only on object
-- add in pathways

drop table if exists com_node_edge_backup;
create table comb_node_edge_backup as select * from comb_node_edge;

-- SCRATCH ---
-- sample p_value query
    -- # the data return order is:
    -- # edge_id
    -- # source ontology code
    -- # target ontology code
    -- # score
    -- # score_type
    -- # source name
    -- # target name
    -- # edge type
    -- # source type
    -- # target type

select ed.edge_id, so.ontology_id, ta.ontology_id, score, sco_type.type_name, so.node_name, ta.node_name, ted.type_name, tso.type_name, tta.type_name
from comb_node_edge ed, comb_node_ontology so, comb_node_ontology ta, comb_lookup_type ted, comb_lookup_type tso, comb_lookup_type tta, comb_lookup_type sco_type 
where ed.source_code = so.node_code and ed.target_code = ta.node_code and ed.edge_type_id = ted.type_id and so.node_type_id = tso.type_id and ta.node_type_id = tta.type_id 
and ed.score_type_id = sco_type.type_id and ed.source_type_id = so.node_type_id and ed.target_type_id = ta.node_type_id and sco_type.type_name = 'biolink:probability'
order by score desc limit 10;


select ed.* from comb_node_edge ed, comb_node_ontology so where ed.source_code = so.node_code and so.ontology_id = 'NCBIGene:1803';

-- edges by combination type
select count(ed.edge_id), sco_type.type_name as score_type, ted.type_name as edge, tso.type_name as source, tta.type_name as target, study.study_name
from comb_node_edge ed, comb_node_ontology so, comb_node_ontology ta, comb_lookup_type ted, comb_lookup_type tso, comb_lookup_type tta, comb_lookup_type sco_type, comb_study_type study
where ed.source_code = so.node_code and ed.target_code = ta.node_code and ed.edge_type_id = ted.type_id and so.node_type_id = tso.type_id and ta.node_type_id = tta.type_id 
and ed.score_type_id = sco_type.type_id and ed.source_type_id = so.node_type_id and ed.target_type_id = ta.node_type_id and ed.study_id = study.study_id
group by sco_type.type_name, ted.type_name, tso.type_name, tta.type_name, study.study_name;

select * 
from comb_node_edge ed, comb_node_ontology so, comb_node_ontology ta, comb_lookup_type ted, comb_lookup_type tso, comb_lookup_type tta, comb_lookup_type sco_type 
where ed.source_code = so.node_code and ed.target_code = ta.node_code and ed.edge_type_id = ted.type_id and so.node_type_id = tso.type_id and ta.node_type_id = tta.type_id 
and ed.score_type_id = sco_type.type_id and tso.type_name = 'biolink:Disease' and tta.type_name = 'biolink:Gene';

-- edge by id
select concat(ed.edge_id, so.ontology_id, ta.ontology_id), so.ontology_id, ta.ontology_id, ed.score, sco_type.type_name, 
so.node_name, ta.node_name, ted.type_name, tso.type_name, tta.type_name         
from comb_node_edge ed, comb_node_ontology so, comb_node_ontology ta, comb_lookup_type ted, comb_lookup_type tso, comb_lookup_type tta, comb_lookup_type sco_type         
where ed.source_code = so.node_code and ed.target_code = ta.node_code and ed.edge_type_id = ted.type_id and so.node_type_id = tso.type_id and ta.node_type_id = tta.type_id         
and ed.score_type_id = sco_type.type_id and ed.source_type_id = so.node_type_id and ed.target_type_id = ta.node_type_id  
and ed.edge_id = 67587;

-- and ted.type_name = %s  and tso.type_name = %s  and tta.type_name = %s  and sco_type.type_name = %s  and so.ontology_id = %s  
-- order by ed.score limit 5000

-- Richards phenotypes
-- 
-- mysql> select distinct target_code from comb_node_edge;
-- +-------------+
-- | target_code |
-- +-------------+
-- | dbilirubin  |
-- | ldl         |
-- | ebmd        |
-- | glucose     |
-- | dbp         |
-- | lowtsh      |
-- | calcium     |
-- | rbc         |
-- | t2d         |
-- | sbp         |
-- | tg          |
-- | height      |
-- +-------------+
-- 12 rows in set (0.02 sec)



-- scratch queries
select count(id) from comb_node_edge where source_code = 'PPARG' and target_code = 'BMI' and source_type_id = 2 and target_type_id in (1, 3, 12);

insert into comb_node_edge (edge_id, edge_type_id, source_code, source_type_id, target_code, target_type_id, score, score_type_id, study_id)
values('test', 5, 'PPARG', 2, 'BreakfastSkipping', 12, 12.0, 8, 1);

insert into comb_node_edge (edge_id, edge_type_id, target_code, target_type_id, source_code, source_type_id, score, score_type_id, study_id)
values('test', 10, 'PPARG', 2, 'BreakfastSkipping', (select node_type_id from comb_node_ontology where node_code = 'BreakfastSkipping' and node_type_id in (1, 3, 12)), 12.0, 8, 1);

update comb_node_edge set score = 20.0 
where (source_code = 'PPARG' and target_code = 'BreakfastSkipping' and source_type_id = 2 and target_type_id in (1, 3, 12))
or (target_code = 'PPARG' and source_code = 'BreakfastSkipping' and target_type_id = 2 and source_type_id in (1, 3, 12))


-- select for gene
select ed.source_code, ed.target_code, ed.score, ot.node_name, ot.ontology_id, ot.node_type_id
from comb_node_edge ed, comb_node_ontology ot
where ed.source_code = 'A2M' and ed.source_type_id = 2 and ed.target_type_id in (1, 3, 12) and ed.score_type_id = 8 
and ed.target_code = ot.node_code and ed.target_type_id = ot.node_type_id;
and ed.score < 0.0000025;



-- updates for disease/phenotypes
update comb_node_ontology set node_type_id = 3, ontology_id = 'EFO:0004308', ontology_type_id = 3 where node_code = 'WBC' and node_type_id = 12;
update comb_node_edge set source_type_id = 3 where source_code = 'WBC' and source_type_id = 12;
update comb_node_edge set target_type_id = 3 where target_code = 'WBC' and target_type_id = 12;

select * from comb_node_ontology where ontology_id is null and node_type_id in (1, 3, 12) order by node_code;

update comb_node_ontology set node_type_id = 3, ontology_id = 'EFO:0010934', ontology_type_id = 3 where id = 44500;
update comb_node_ontology set node_type_id = 3, ontology_id = 'EFO:0004842', ontology_type_id = 3 where id = 44495;
update comb_node_ontology set node_type_id = 3, ontology_id = 'UMLS:C0556228', ontology_type_id = 5 where id = 44458;

update comb_node_ontology set node_type_id = 3, ontology_id = 'HP:0025267', ontology_type_id = 6 where id = 44329;
update comb_node_ontology set node_type_id = 3, ontology_id = 'MONDO:0008638', ontology_type_id = 2 where id = 44320;
update comb_node_ontology set node_type_id = 3, ontology_id = 'EFO:0009884', ontology_type_id = 3 where id = 44464;
update comb_node_ontology set node_type_id = 3, ontology_id = 'EFO:0009883', ontology_type_id = 3 where id = 44465;
update comb_node_ontology set node_type_id = 3, ontology_id = 'EFO:0009882', ontology_type_id = 3 where id = 44466;
update comb_node_ontology set node_type_id = 1, ontology_id = 'MONDO:0005098', ontology_type_id = 2 where id = 44405;
update comb_node_ontology set node_type_id = 1, ontology_id = 'MONDO:0020671', ontology_type_id = 2 where id = 44404;


update comb_node_ontology set node_type_id = 1, ontology_id = 'MONDO:0002025', ontology_type_id = 2 where id = 44318;
update comb_node_ontology set node_type_id = 3, ontology_id = 'EFO:0003923', ontology_type_id = 3 where id = 44467;
update comb_node_ontology set node_type_id = 1, ontology_id = 'MONDO:0002525', ontology_type_id = 2 where id = 44311;
update comb_node_ontology set node_type_id = 1, ontology_id = 'MONDO:0004678', ontology_type_id = 2 where id = 44310;
update comb_node_ontology set node_type_id = 3, ontology_id = 'EFO:0004348', ontology_type_id = 3 where id = 44485;
update comb_node_ontology set node_type_id = 1, ontology_id = 'MONDO:0004872', ontology_type_id = 2 where id = 44312;
update comb_node_ontology set node_type_id = 3, ontology_id = 'UMLS:C3150710', ontology_type_id = 5 where id = 44360;
update comb_node_ontology set node_type_id = 1, ontology_id = 'MONDO:0011786', ontology_type_id = 2 where id = 44312;



update comb_node_ontology set node_type_id = 3, ontology_id = 'EFO:0009883', ontology_type_id = 3 where id = 44465;
update comb_node_ontology set node_type_id = 3, ontology_id = 'EFO:0009883', ontology_type_id = 3 where id = 44465;


-- NOTTES
-- add ontology type
-- add data to edge
-- add ontology_id for diseases

-- deleting extra disease nodes
select a.id, b.id, a.node_code, b.node_code, a.ontology_id, b.ontology_id from comb_node_ontology a, comb_node_ontology b 
where b.node_code = a .node_code and b.id != a.id and b.node_type_id = a.node_type_id and a.ontology_type_id = 3;

delete from comb_node_ontology where id in 
(
32832
, 32833
, 32834
, 32835
, 32836
, 32837
, 32838
, 32839
, 32840
, 32841
, 32842
, 32843
, 32844
, 32845
, 32846
, 32847
, 32849
, 32850
, 32852
, 32853
, 32854
, 32855
, 32856
, 32857
, 32858
, 32859
, 32860
, 32861
, 32862
, 32863
, 32864
, 32865
, 32895
)

