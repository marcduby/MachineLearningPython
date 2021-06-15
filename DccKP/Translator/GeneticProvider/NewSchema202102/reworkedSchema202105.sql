

-- old sql to account for multiple curies for one phenotype/disease (not needed with node normalizer)
select concat(ed.edge_id, so.ontology_id, ta.ontology_id), so.ontology_id, ta.ontology_id, ed.score, sco_type.type_name, so.node_name, ta.node_name, ted.type_name, tso.type_name, tta.type_name \
from comb_node_edge ed, comb_node_ontology so, comb_node_ontology ta, comb_lookup_type ted, comb_lookup_type tso, comb_lookup_type tta, comb_lookup_type sco_type \
where ed.source_code = so.node_code 
and ed.source_type_id = so.node_type_id 
and ed.target_code = ta.node_code 
and ed.target_type_id = ta.node_type_id

and ed.edge_type_id = ted.type_id 
and so.node_type_id = tso.type_id 
and ta.node_type_id = tta.type_id \
and ed.score_type_id = sco_type.type_id 

-- 1 - define new edge table
drop table if exists comb_edge_node;
create table comb_edge_node (
  id                        int not null auto_increment primary key,
  edge_id                   varchar(100) not null,
  source_node_id            int(3) not null,
  target_node_id            int(3) not null,
  edge_type_id              int(3) not null,
  score                     double,
  score_text                varchar(50),
  score_type_id             int(3) not null,
  study_id                  int(3) not null,
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
);
alter table comb_edge_node add index comb_edg_nod_src_idx (source_node_id);
alter table comb_edge_node add index comb_edg_nod_tgt_idx (target_node_id);
alter table comb_edge_node add index comb_edg_nod_sco_idx (score);
alter table comb_edge_node add index comb_edg_nod_sco_typ_idx (score_type_id);


-- alter node ontology table
alter table comb_node_ontology add last_updated DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP;


-- 1a - delete duplicate phenotypes
create table comb_node_ontology_20210524 select * from comb_node_ontology;
delete from comb_node_ontology where id in (
select a.id 
from comb_node_ontology_20210524 a, comb_node_ontology_20210524 b 
where b.node_code = a .node_code and b.id != a.id and b.node_type_id = a.node_type_id and a.ontology_type_id = 1);

-- 165704 rows
select distinct source_node_id from comb_edge_node where source_node_id not in (select id from comb_node_ontology);
select distinct target_node_id from comb_edge_node where target_node_id not in (select id from comb_node_ontology);

create table tran_curie as 
select a.id as new_id, b.id as old_id
from comb_node_ontology_20210524 a, comb_node_ontology_20210524 b 
where b.node_code = a .node_code and b.id != a.id and b.node_type_id = a.node_type_id and a.ontology_type_id = 2;

select ed.source_node_id
from comb_edge_node ed, tran_curie tr 
where ed.source_node_id = tr.old_id

-- use temp table
update comb_edge_node ed
inner join tran_curie tr  
on ed.source_node_id = tr.old_id
set ed.source_node_id = tr.new_id;

update comb_edge_node ed
inner join tran_curie tr  
on ed.target_node_id = tr.old_id
set ed.target_node_id = tr.new_id;

-- 2 - load magna
-- load phenoptypes into new table
-- 20210519 - make ontology_id nullable for adding in new phenotypes
alter table comb_node_ontology modify ontology_id varchar(50) null;
alter table comb_node_ontology modify ontology_type_id varchar(50) null;

-- load from aggregator

-- 3 - load clinvar unique results
insert into comb_edge_node (edge_id, source_node_id, target_node_id, edge_type_id, score, score_text, score_type_id, study_id)
select min(ed.edge_id)as tot, so.id, ta.id, ed.edge_type_id, ed.score, ed.score_text, ed.score_type_id, ed.study_id
from comb_node_edge ed, comb_node_ontology so, comb_node_ontology ta
where ed.source_code = so.node_code 
and ed.source_type_id = so.node_type_id 
and ed.target_code = ta.node_code 
and ed.target_type_id = ta.node_type_id
and study_id = 5
group by so.id, ta.id, ed.edge_type_id, ed.score, ed.score_text, ed.score_type_id, ed.study_id
order by tot;

insert into comb_edge_node (edge_id, source_node_id, target_node_id, edge_type_id, score, score_text, score_type_id, study_id)
select min(ed.edge_id)as tot, so.id, ta.id, ed.edge_type_id, ed.score, ed.score_text, ed.score_type_id, ed.study_id
from comb_node_edge ed, comb_node_ontology so, comb_node_ontology ta
where ed.source_code = so.node_code 
and ed.source_type_id = so.node_type_id 
and ed.target_code = ta.node_code 
and ed.target_type_id = ta.node_type_id
and study_id = 6
group by so.id, ta.id, ed.edge_type_id, ed.score, ed.score_text, ed.score_type_id, ed.study_id
order by tot;


-- 4 - load richards data
insert into comb_edge_node (edge_id, source_node_id, target_node_id, edge_type_id, score, score_text, score_type_id, study_id)
select ed.edge_id, so.id, ta.id, ed.edge_type_id, ed.score, ed.score_text, ed.score_type_id, ed.study_id
from comb_node_edge ed, comb_node_ontology so, comb_node_ontology ta
where ed.source_code = so.node_code 
and ed.source_type_id = so.node_type_id 
and ed.target_code = ta.node_code 
and ed.target_type_id = ta.node_type_id
and study_id = 4;

-- 5 - load magma pathways
insert into comb_edge_node (edge_id, source_node_id, target_node_id, edge_type_id, score, score_text, score_type_id, study_id)
select ed.edge_id, so.id, ta.id, ed.edge_type_id, ed.score, ed.score_text, ed.score_type_id, ed.study_id
from comb_node_edge ed, comb_node_ontology so, comb_node_ontology ta
where ed.source_code = so.node_code 
and ed.source_type_id = so.node_type_id 
and ed.target_code = ta.node_code 
and ed.target_type_id = ta.node_type_id
and study_id = 1 and (ed.target_type_id = 4 or ed.source_type_id = 4);




-- scratch
select * from comb_node_ontology where ontology_id is null;
SELECT * FROM comb_node_ontology WHERE last_updated >= NOW() - INTERVAL 2 MINUTE;

ONDO:0004979

-- done
-- | 44406 | Asthma                                   |           12 | NULL        | NULL             | Asthma                                                                             |
update comb_node_ontology set ontology_id = 'MONDO:0004979', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:Disease'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'MONDO%') 
where id = 44406 and node_code = 'Asthma';

-- | 44414 | Osteoporosis                             |           12 | NULL        | NULL             | Osteoporosis                                                                       |
update comb_node_ontology set ontology_id = 'MONDO:0005298', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:Disease'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'MONDO%') 
where id = 44414 and node_code = 'Osteoporosis';

-- | 44405 | AllergicRhinitis                         |           12 | NULL        | NULL             | Allergic rhinitis                                                                  |
update comb_node_ontology set ontology_id = 'MONDO:0011786', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:Disease'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'MONDO%') 
where id = 44405 and node_code = 'AllergicRhinitis';

-- | 44437 | ALBUMIN                                  |           12 | NULL        | NULL             | Urinary albumin                                                                    |
update comb_node_ontology set ontology_id = 'EFO:0004285', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:PhenotypicFeature'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'EFO%') 
where id = 44437 and node_code = 'ALBUMIN';

-- | 44442 | AnyCVD                                   |           12 | NULL        | NULL             | Any cardiovascular disease                                                         |
update comb_node_ontology set ontology_id = 'MONDO:0004995', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:Disease'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'MONDO%') 
where id = 44442 and node_code = 'AnyCVD';

-- | 44592 | BasoCount                                |           12 | NULL        | NULL             | Basophil count                                                                     |
update comb_node_ontology set ontology_id = 'EFO:0005090', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:PhenotypicFeature'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'EFO%') 
where id = 44592 and node_code = 'BasoCount';

-- | 44504 | allSVS                                   |           12 | NULL        | NULL             | Intracerebral hemorrhage or small vessel ischemic stroke                           |
update comb_node_ontology set ontology_id = 'MONDO:0013792', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:Disease'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'MONDO%') 
where id = 44504 and node_code = 'allSVS';

-- | 44619 | BreakfastSkipping                        |           12 | NULL        | NULL             | Frequency of breakfast skipping                                                    |
update comb_node_ontology set ontology_id = 'EFO:0010129', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:PhenotypicFeature'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'EFO%') 
where id = 44619 and node_code = 'BreakfastSkipping';

-- | 44469 | ChildObesity                             |           12 | NULL        | NULL             | Childhood obesity                                                                  |
update comb_node_ontology set ontology_id = 'NCIT:C120377', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:PhenotypicFeature'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'NCIT%') 
where id = 44469 and node_code = 'ChildObesity';

-- | 44407 | Cancer                                   |           12 | NULL        | NULL             | Any cancer                                                                         |
update comb_node_ontology set ontology_id = 'MONDO:0004992', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:Disease'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'MONDO%') 
where id = 44407 and node_code = 'Cancer';

-- | 44408 | Dermatophytosis                          |           12 | NULL        | NULL             | Dermatophytosis                                                                    |
update comb_node_ontology set ontology_id = 'MONDO:0005982', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:Disease'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'MONDO%') 
where id = 44408 and node_code = 'Dermatophytosis';

-- | 44409 | Dyslipid                                 |           12 | NULL        | NULL             | Dyslipidemia                                                                       |
update comb_node_ontology set ontology_id = 'MONDO:0002525', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:Disease'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'MONDO%') 
where id = 44409 and node_code = 'Dyslipid';

-- | 44593 | EosinCount                               |           12 | NULL        | NULL             | Eosinophil count                                                                   |
update comb_node_ontology set ontology_id = 'EFO:0004842', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:PhenotypicFeature'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'EFO%') 
where id = 44593 and node_code = 'EosinCount';

-- | 44556 | freshfruit                               |           12 | NULL        | NULL             | Fresh fruit consumption                                                            |
update comb_node_ontology set ontology_id = 'UMLS:C0556228', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:PhenotypicFeature'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'UMLS%') 
where id = 44556 and node_code = 'freshfruit';

-- | 44485 | GA                                       |           12 | NULL        | NULL             | Geographic atrophy                                                                 |
update comb_node_ontology set ontology_id = 'HP:0031609', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:PhenotypicFeature'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'HP%') 
where id = 44485 and node_code = 'GA';

-- | 44583 | Hematocrit                               |           12 | NULL        | NULL             | Hematocrit                                                                         |
update comb_node_ontology set ontology_id = 'EFO:0004348', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:PhenotypicFeature'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'EFO%') 
where id = 44583 and node_code = 'Hematocrit';

-- | 44410 | Hemorrhoids                              |           12 | NULL        | NULL             | Hemorrhoids                                                                        |
update comb_node_ontology set ontology_id = 'MONDO:0004872', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:Disease'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'MONDO%') 
where id = 44410 and node_code = 'Hemorrhoids';

-- | 44413 | IronDef                                  |           12 | NULL        | NULL             | Iron deficiency                                                                    |
update comb_node_ontology set ontology_id = 'MONDO:0001356', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:PhenotypicFeature'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'MONDO%') 
where id = 44413 and node_code = 'IronDef';

-- | 44436 | IPF                                      |           12 | NULL        | NULL             | Idiopathic pulmonary fibrosis                                                      |
update comb_node_ontology set ontology_id = 'MONDO:0008345', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:Disease'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'MONDO%') 
where id = 44436 and node_code = 'IPF';

-- | 44447 | HIPKNEEOA                                |           12 | NULL        | NULL             | Knee and-or hip osteoarthritis                                                     |
update comb_node_ontology set ontology_id = 'MONDO:0011923', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:Disease'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'MONDO%') 
where id = 44447 and node_code = 'HIPKNEEOA';

-- | 44412 | IBS                                      |           12 | NULL        | NULL             | Irritable bowel syndrome                                                           |
update comb_node_ontology set ontology_id = 'MONDO:0005052', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:Disease'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'MONDO%') 
where id = 44412 and node_code = 'IBS';

-- | 44543 | ISR                                      |           12 | NULL        | NULL             | Insulin secretion rate                                                             |
update comb_node_ontology set ontology_id = 'EFO:0008001', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:PhenotypicFeature'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'EFO%') 
where id = 44543 and node_code = 'ISR';

-- | 44505 | lobarSVS                                 |           12 | NULL        | NULL             | Lobar intracerebral hemorrhage or small vessel ischemic stroke                     |
-- update comb_node_ontology set ontology_id = 'EFO:0010177', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:PhenotypicFeature'), 
-- ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'EFO%') 
-- where id = 44505 and node_code = 'lobarSVS';

-- | 44590 | LymphoCount                              |           12 | NULL        | NULL             | Lymphocyte count                                                                   |
update comb_node_ontology set ontology_id = 'EFO:0004587', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:PhenotypicFeature'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'EFO%') 
where id = 44590 and node_code = 'LymphoCount';

-- | 44496 | MA                                       |           12 | NULL        | NULL             | Microalbuminuria                                                                   |
update comb_node_ontology set ontology_id = 'HP:0012594', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:PhenotypicFeature'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'HP%') 
where id = 44496 and node_code = 'MA';

-- | 44584 | MeanCorpHb                               |           12 | NULL        | NULL             | Mean corpuscular hemoglobin                                                        |
update comb_node_ontology set ontology_id = 'EFO:0004527', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:PhenotypicFeature'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'EFO%') 
where id = 44584 and node_code = 'MeanCorpHb';

-- | 44585 | MeanCorpVol                              |           12 | NULL        | NULL             | Mean corpuscular volume                                                            |
update comb_node_ontology set ontology_id = 'EFO:0004527', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:PhenotypicFeature'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'EFO%') 
where id = 44585 and node_code = 'MeanCorpVol';

-- | 44608 | MicroInT2D                               |           12 | NULL        | NULL             | Microalbuminuria in type 2 diabetes                                                |
update comb_node_ontology set ontology_id = 'UMLS:C3875084', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:PhenotypicFeature'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'UMLS%') 
where id = 44608 and node_code = 'MicroInT2D';

-- | 44591 | MonoCount                                |           12 | NULL        | NULL             | Monocyte count                                                                     |
update comb_node_ontology set ontology_id = 'EFO:0005091', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:PhenotypicFeature'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'EFO%') 
where id = 44591 and node_code = 'MonoCount';

-- | 44612 | LongQT                                   |           12 | NULL        | NULL             | Long QT syndrome                                                                   |
update comb_node_ontology set ontology_id = 'MONDO:0002442', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:Disease'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'MONDO%') 
where id = 44612 and node_code = 'LongQT';

-- | 44586 | MeanCorpHbConc                           |           12 | NULL        | NULL             | Mean corpuscular hemoglobin concentration                                          |
update comb_node_ontology set ontology_id = 'EFO:0004528', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:PhenotypicFeature'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'EFO%') 
where id = 44586 and node_code = 'MeanCorpHbConc';

-- | 44589 | NeutCount                                |           12 | NULL        | NULL             | Neutrophil count                                                                   |
update comb_node_ontology set ontology_id = 'EFO:0004833', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:PhenotypicFeature'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'EFO%') 
where id = 44589 and node_code = 'NeutCount';

-- | 44557 | oilyfish                                 |           12 | NULL        | NULL             | Oily fish consumption                                                              |
update comb_node_ontology set ontology_id = 'UMLS:C0556218', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:PhenotypicFeature'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'UMLS%') 
where id = 44557 and node_code = 'oilyfish';

-- | 44561 | PepticUlcers                             |           12 | NULL        | NULL             | Peptic ulcers                                                                      |
update comb_node_ontology set ontology_id = 'MONDO:0004247', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:Disease'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'MONDO%') 
where id = 44561 and node_code = 'PepticUlcers';

-- | 44594 | PlatCount                                |           12 | NULL        | NULL             | Platelet count                                                                     |
update comb_node_ontology set ontology_id = 'EFO:0004309', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:PhenotypicFeature'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'EFO%') 
where id = 44594 and node_code = 'PlatCount';

-- | 44595 | PlatVol                                  |           12 | NULL        | NULL             | Mean platelet volume                                                               |
update comb_node_ontology set ontology_id = 'EFO:0004584', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:PhenotypicFeature'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'EFO%') 
where id = 44595 and node_code = 'PlatVol';

-- | 44415 | PVD                                      |           12 | NULL        | NULL             | Peripheral vascular disease                                                        |
update comb_node_ontology set ontology_id = 'MONDO:0005294', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:Disease'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'MONDO%') 
where id = 44415 and node_code = 'PVD';

-- | 44416 | Psychiatric                              |           12 | NULL        | NULL             | Psychiatric disorders                                                              |
update comb_node_ontology set ontology_id = 'MONDO:0005084', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:Disease'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'MONDO%') 
where id = 44416 and node_code = 'Psychiatric';

-- | 44427 | Snoring                                  |           12 | NULL        | NULL             | Snoring                                                                            |
update comb_node_ontology set ontology_id = 'HP:0025267', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:PhenotypicFeature'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'HP%') 
where id = 44427 and node_code = 'Snoring';

-- | 44502 | Stroke_ischemic                          |           12 | NULL        | NULL             | All ischemic stroke                                                                |
update comb_node_ontology set ontology_id = 'MONDO:0020671', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:Disease'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'MONDO%') 
where id = 44502 and node_code = 'Stroke_ischemic';

-- | 44503 | Stroke                                   |           12 | NULL        | NULL             | Any stroke                                                                         |
update comb_node_ontology set ontology_id = 'MONDO:0005098', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:Disease'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'MONDO%') 
where id = 44503 and node_code = 'Stroke';

-- | 44417 | Stress                                   |           12 | NULL        | NULL             | Acute reaction to stress                                                           |
update comb_node_ontology set ontology_id = 'MONDO:0003763', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:Disease'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'MONDO%') 
where id = 44417 and node_code = 'Stress';

-- | 44565 | TB-BMD                                   |           12 | NULL        | NULL             | Total body bone mineral density                                                    |
update comb_node_ontology set ontology_id = 'EFO:0003923', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:PhenotypicFeature'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'EFO%') 
where id = 44565 and node_code = 'TB-BMD';

-- | 44564 | UPCR                                     |           12 | NULL        | NULL             | Urinary potassium-to-creatinine ratio                                              |
update comb_node_ontology set ontology_id = 'EFO:0009882', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:PhenotypicFeature'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'EFO%') 
where id = 44564 and node_code = 'UPCR';

-- | 44563 | USCR                                     |           12 | NULL        | NULL             | Urinary sodium-to-creatinine ratio                                                 |
update comb_node_ontology set ontology_id = 'EFO:0009883', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:PhenotypicFeature'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'EFO%') 
where id = 44563 and node_code = 'USCR';

-- | 44562 | USPR                                     |           12 | NULL        | NULL             | Urinary sodium-to-potassium ratio                                                  |
update comb_node_ontology set ontology_id = 'EFO:0009884', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:PhenotypicFeature'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'EFO%') 
where id = 44562 and node_code = 'USPR';

-- | 44418 | VaricoseVeins                            |           12 | NULL        | NULL             | Varicose veins                                                                     |
update comb_node_ontology set ontology_id = 'MONDO:0008638', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:Disease'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'MONDO%') 
where id = 44418 and node_code = 'VaricoseVeins';

-- | 44537 | VLDLchol                                 |           12 | NULL        | NULL             | VLDL cholesterol                                                                   |
update comb_node_ontology set ontology_id = 'EFO:0008317', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:PhenotypicFeature'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'EFO%') 
where id = 44537 and node_code = 'VLDLchol';

-- | 44588 | WBC                                      |           12 | NULL        | NULL             | White blood cell count                                                             |
update comb_node_ontology set ontology_id = 'EFO:0004308', node_type_id = (select type_id from comb_lookup_type where type_name = 'biolink:PhenotypicFeature'), 
ontology_type_id = (select ontology_id from comb_ontology_type where ontology_name like 'EFO%') 
where id = 44588 and node_code = 'WBC';



mysql> select * from comb_node_ontology where ontology_id is null and node_type_id in (1, 3, 12) order by node_code;
+-------+------------------------------------------+--------------+-------------+------------------+------------------------------------------------------------------------------------+
| id    | node_code                                | node_type_id | ontology_id | ontology_type_id | node_name                                                                          |
+-------+------------------------------------------+--------------+-------------+------------------+------------------------------------------------------------------------------------+
+-------+------------------------------------------+--------------+-------------+------------------+------------------------------------------------------------------------------------+


+-------+------------------------------------------+--------------+-------------+------------------+------------------------------------------------------------------------------------+
| id    | node_code                                | node_type_id | ontology_id | ontology_type_id | node_name                                                                          |
+-------+------------------------------------------+--------------+-------------+------------------+------------------------------------------------------------------------------------+
| 44539 | 2hrCPEP                                  |           12 | NULL        | NULL             | Two-hour C-peptide                                                                 |
| 44525 | 2hrFFA                                   |           12 | NULL        | NULL             | 2hr plasma free fatty acids                                                        |
| 44458 | 2hrG                                     |           12 | NULL        | NULL             | Two-hour glucose                                                                   |
| 44459 | 2hrI                                     |           12 | NULL        | NULL             | Two-hour insulin                                                                   |
| 44422 | AFR                                      |           12 | NULL        | NULL             | Arm fat ratio                                                                      |
| 44460 | AFxAGE                                   |           12 | NULL        | NULL             | AF-SNP age interaction                                                             |
| 44461 | AFxAGEo65                                |           12 | NULL        | NULL             | AF-over age 65                                                                     |
| 44462 | AFxAGEy65                                |           12 | NULL        | NULL             | AF-age 65 and under                                                                |
| 44463 | AFxBMI                                   |           12 | NULL        | NULL             | AF-SNP BMI interaction                                                             |
| 44464 | AFxHTN                                   |           12 | NULL        | NULL             | AF-SNP hypertension interaction                                                    |
| 44465 | AFxSEX                                   |           12 | NULL        | NULL             | AF-SNP sex interaction                                                             |
| 44542 | AIRadjBMInSI                             |           12 | NULL        | NULL             | Acute insulin response adj BMI-SI                                                  |
| 44546 | AIRadjSI                                 |           12 | NULL        | NULL             | Acute insulin response adj SI                                                      |
| 44599 | AlbInT2D                                 |           12 | NULL        | NULL             | Albuminuria in type 2 diabetes                                                     |
| 44466 | allDKDadjHbA1cBMI                        |           12 | NULL        | NULL             | All diabetic kidney disease adj HbA1c-BMI                                          |
| 44603 | AnyCVDinT2D                              |           12 | NULL        | NULL             | Cardiovascular disease in type 2 diabetes                                          |
| 44610 | AscAortaDiam                             |           12 | NULL        | NULL             | Ascending aorta diameter                                                           |
| 44598 | AST_ALT_ratio                            |           12 | NULL        | NULL             | AST-ALT ratio                                                                      |
| 44467 | AUCins                                   |           12 | NULL        | NULL             | Area under the curve (AUC) for insulin                                             |
| 44468 | AUCinsAUCgluc                            |           12 | NULL        | NULL             | AUCins over AUCgluc                                                                |
| 44613 | BMB_any_noexclusions                     |           12 | NULL        | NULL             | Brain microbleeds, any                                                             |
| 44615 | BMB_any_withexclusions                   |           12 | NULL        | NULL             | Brain microbleeds, any, excluding dementia and stroke cases                        |
| 44617 | BMB_mixed_or_strictlydeep_noexclusions   |           12 | NULL        | NULL             | Brain microbleeds, mixed or strictly deep                                          |
| 44614 | BMB_mixed_or_strictlydeep_withexclusions |           12 | NULL        | NULL             | Brain microbleeds, mixed or strictly deep, excluding dementia and stroke cases     |
| 44616 | BMB_strictlylobar_noexclusions           |           12 | NULL        | NULL             | Brain microbleeds, strictly lobar                                                  |
| 44618 | BMB_strictlylobar_withexclusions         |           12 | NULL        | NULL             | Brain microbleeds, strictly lobar, excluding dementia and stroke cases             |
| 44553 | BMIadjSMK                                |           12 | NULL        | NULL             | BMI adj smoking status                                                             |
| 44514 | BS                                       |           12 | NULL        | NULL             | Random glucose                                                                     |
| 44551 | CADinNonT2D                              |           12 | NULL        | NULL             | Coronary artery disease in subjects without diabetes                               |
| 44580 | CADinT1D                                 |           12 | NULL        | NULL             | Coronary artery disease in type 1 diabetes                                         |
| 44550 | CADinT2D                                 |           12 | NULL        | NULL             | Coronary artery disease in type 2 diabetes                                         |
| 44398 | ChildSleepDuration                       |           12 | NULL        | NULL             | Sleep duration in children                                                         |
| 44443 | ChildSleepDurationAdjBMI                 |           12 | NULL        | NULL             | Sleep duration in children adj BMI                                                 |
| 44569 | ChronotypeSingle                         |           12 | NULL        | NULL             | Chronotype (single question)                                                       |
| 44470 | CIR                                      |           12 | NULL        | NULL             | Corrected insulin response (CIR)                                                   |
| 44471 | CIRadjISI                                |           12 | NULL        | NULL             | Corrected insulin response adj Matsuda ISI                                         |
| 44601 | CKD_DNinT2D                              |           12 | NULL        | NULL             | Chronic kidney disease and diabetic nephropathy in type 2 diabetes                 |
| 44602 | CKD_EXT_vGFRinT2D                        |           12 | NULL        | NULL             | Extreme chronic kidney disease vs. eGFR >= 60 in type 2 diabetes                   |
| 44472 | CKDadjHbA1cBMI                           |           12 | NULL        | NULL             | Chronic kidney disease adj HbA1c-BMI                                               |
| 44473 | CKDextremes                              |           12 | NULL        | NULL             | Extreme chronic kidney disease                                                     |
| 44474 | CKDextremesadjHbA1cBMI                   |           12 | NULL        | NULL             | Extreme chronic kidney disease adj HbA1c-BMI                                       |
| 44526 | CKDinT2D                                 |           12 | NULL        | NULL             | Chronic kidney disease in type 2 diabetes                                          |
| 44475 | CKDpDKD                                  |           12 | NULL        | NULL             | CKD and DKD                                                                        |
| 44476 | CKDpDKDadjHbA1cBMI                       |           12 | NULL        | NULL             | Chronic kidney disease and diabetic kidney disease adj HbA1c-BMI                   |
| 44600 | CKDpDKDinT2D                             |           12 | NULL        | NULL             | CKD and DKD in type 2 diabetes                                                     |
| 44444 | ClaudicationinT2D                        |           12 | NULL        | NULL             | Claudication in type 2 diabetes                                                    |
| 44547 | CreatinineUrinary                        |           12 | NULL        | NULL             | Urinary creatinine                                                                 |
| 44552 | CVDinT2D                                 |           12 | NULL        | NULL             | Coronary heart disease or stroke or peripheral vascular disease in type 2 diabetes |
| 44611 | DescAortaDiam                            |           12 | NULL        | NULL             | Descending aorta diameter                                                          |
| 44527 | DHA                                      |           12 | NULL        | NULL             | Dicosahexaneoic acid                                                               |
| 44540 | DIadjBMI                                 |           12 | NULL        | NULL             | Disposition index adj BMI                                                          |
| 44445 | DRvNoDR                                  |           12 | NULL        | NULL             | Diabetic retinopathy (DR) vs no DR                                                 |
| 44446 | DRvNoDRwDoD                              |           12 | NULL        | NULL             | Diabetic retinopathy (DR) vs no DR acct DoD-GC                                     |
| 44429 | EaseOfWakingUp                           |           12 | NULL        | NULL             | Ease of waking up                                                                  |
| 44477 | eGFRcrea                                 |           12 | NULL        | NULL             | eGFR-creat (serum creatinine)                                                      |
| 44478 | eGFRcys                                  |           12 | NULL        | NULL             | eGFR-cys (serum cystatin C)                                                        |
| 44528 | ESRDinT2D                                |           12 | NULL        | NULL             | End-stage renal disease in type 2 diabetes                                         |
| 44479 | ESRDvControl                             |           12 | NULL        | NULL             | End-stage renal disease vs. controls                                               |
| 44480 | ESRDvControladjHbA1cBMI                  |           12 | NULL        | NULL             | End-stage renal disease vs. controls adj HbA1c-BMI                                 |
| 44604 | ESRDvControlinT2D                        |           12 | NULL        | NULL             | End-stage renal disease vs. controls in type 2 diabetes                            |
| 44481 | ESRDvMacro                               |           12 | NULL        | NULL             | End-stage renal disease vs. macroalbuminuria                                       |
| 44482 | ESRDvMacroadjHbA1cBMI                    |           12 | NULL        | NULL             | End-stage renal disease vs. macroalbuminuria adj HbA1c-BMI                         |
| 44605 | ESRDvMacroinT2D                          |           12 | NULL        | NULL             | End-stage renal disease vs. macroalbuminuria in type 2 diabetes                    |
| 44483 | ESRDvNonESRD                             |           12 | NULL        | NULL             | End-stage renal disease vs. no ESRD                                                |
| 44484 | ESRDvNonESRDadjHbA1cBMI                  |           12 | NULL        | NULL             | End-stage renal disease vs. no ESRD adj HbA1c-BMI                                  |
| 44606 | ESRDvNonESRDinT2D                        |           12 | NULL        | NULL             | End-stage renal disease vs. no ESRD in type 2 diabetes                             |
| 44529 | FastFFA                                  |           12 | NULL        | NULL             | Fasting plasma free fatty acids                                                    |
| 44572 | FGinterBMI                               |           12 | NULL        | NULL             | Fasting glucose-BMI interaction                                                    |
| 44573 | FIinterBMI                               |           12 | NULL        | NULL             | Fasting insulin-BMI interaction                                                    |
| 44568 | ForearmBMD                               |           12 | NULL        | NULL             | Forearm bone mineral density                                                       |
| 44538 | HBA1CadjBMI                              |           12 | NULL        | NULL             | HbA1c adj BMI                                                                      |
| 44438 | HBA1CMMOL                                |           12 | NULL        | NULL             | HbA1c (mmol per mol)                                                               |
| 44582 | HbConc                                   |           12 | NULL        | NULL             | Hemoglobin concentration                                                           |
| 44439 | HDL_RATIO                                |           12 | NULL        | NULL             | Ratio total to HDL cholesterol                                                     |
| 44530 | HDL2chol                                 |           12 | NULL        | NULL             | HDL2 cholesterol                                                                   |
| 44531 | HDL3chol                                 |           12 | NULL        | NULL             | HDL3 cholesterol                                                                   |
| 44411 | HerniaAbdomin                            |           12 | NULL        | NULL             | Hernia abdominopelvic cavity                                                       |
| 44577 | HospC19vAll                              |           12 | NULL        | NULL             | Hospitalized COVID-19 vs population                                                |
| 44576 | HospvNonHospC19                          |           12 | NULL        | NULL             | Hospitalized vs non-hospitalized COVID-19                                          |
| 44448 | HypertensioninT2D                        |           12 | NULL        | NULL             | Hypertension in type 2 diabetes                                                    |
| 44567 | IAU                                      |           12 | NULL        | NULL             | Unruptured intracranial aneurysm                                                   |
| 44532 | IDLpart                                  |           12 | NULL        | NULL             | Concentration of IDL particles                                                     |
| 44486 | Incr30                                   |           12 | NULL        | NULL             | Incremental insulin at 30 min OGTT                                                 |
| 44487 | Ins30                                    |           12 | NULL        | NULL             | Insulin at 30 min OGTT                                                             |
| 44488 | Ins30adjBMI                              |           12 | NULL        | NULL             | Insulin at 30 min OGTT adj BMI                                                     |
| 44489 | ISenBMI                                  |           12 | NULL        | NULL             | Insulin sensitivity adj BMI                                                        |
| 44490 | ISI                                      |           12 | NULL        | NULL             | Matsuda insulin sensitivity index (ISI)                                            |
| 44491 | ISIadjAgeSex                             |           12 | NULL        | NULL             | Modified Stumvoll ISI adj age-sex                                                  |
| 44492 | ISIadjAgeSexBMI                          |           12 | NULL        | NULL             | Modified Stumvoll ISI adj age-sex-BMI                                              |
| 44493 | ISIadjBMIinterGenoBMI                    |           12 | NULL        | NULL             | Modified Stumvoll ISI adj genotype-BMI interaction                                 |
| 44545 | ISRadjBMI                                |           12 | NULL        | NULL             | Insulin secretion rate adj BMI                                                     |
| 44579 | KExcretion                               |           12 | NULL        | NULL             | Urinary potassium excretion                                                        |
| 44494 | lateDKDadjHbA1cBMI                       |           12 | NULL        | NULL             | Late diabetic kidney disease adj HbA1c-BMI                                         |
| 44440 | LDL_CALCULATED                           |           12 | NULL        | NULL             | Calculated LDL cholesterol                                                         |
| 44495 | LEPadjBMI                                |           12 | NULL        | NULL             | Leptin adj BMI                                                                     |
| 44423 | LFR                                      |           12 | NULL        | NULL             | Leg fat ratio                                                                      |
| 44441 | LOG_TG                                   |           12 | NULL        | NULL             | Log triglyceride level                                                             |
| 44596 | LowGripEWGSOP                            |           12 | NULL        | NULL             | Low hand grip strength (EWGSOP definition)                                         |
| 44597 | LowGripFNIH                              |           12 | NULL        | NULL             | Low hand grip strength (FNIH definition)                                           |
| 44548 | LSarea                                   |           12 | NULL        | NULL             | Lumbar spine area                                                                  |
| 44549 | LSBMD                                    |           12 | NULL        | NULL             | Lumbar spine bone mineral density                                                  |
| 44449 | LVEDV                                    |           12 | NULL        | NULL             | Left ventricular end-diastolic volume                                              |
| 44450 | LVEDVI                                   |           12 | NULL        | NULL             | Left ventricular end-diastolic volume (BSA-indexed)                                |
| 44451 | LVESV                                    |           12 | NULL        | NULL             | Left ventricular end-systolic volume                                               |
| 44452 | LVESVI                                   |           12 | NULL        | NULL             | Left ventricular end-systolic volume (BSA-indexed)                                 |
| 44497 | MAadjHbA1cBMI                            |           12 | NULL        | NULL             | Microalbuminuria adj HbA1c-BMI                                                     |
| 44607 | MacroInT2D                               |           12 | NULL        | NULL             | Macroalbuminuria in type 2 diabetes                                                |
| 44498 | MacrovControl                            |           12 | NULL        | NULL             | Macroalbuminuria vs. controls                                                      |
| 44499 | MacrovControladjHbA1cBMI                 |           12 | NULL        | NULL             | Macroalbuminuria vs. controls adj HbA1c-BMI                                        |
| 44518 | MeanSleepDiurnalInactRN                  |           12 | NULL        | NULL             | Diurnal inactivity duration, rank-normalized                                       |
| 44519 | MeanSleepEfficiencyRN                    |           12 | NULL        | NULL             | Sleep efficiency, rank-normalized                                                  |
| 44520 | MeanSleepL5timeRN                        |           12 | NULL        | NULL             | Least-active 5 hour timing, rank-normalized                                        |
| 44521 | MeanSleepM10timeRN                       |           12 | NULL        | NULL             | Most-active 10 hour timing, rank-normalized                                        |
| 44522 | MeanSleepMidPointRN                      |           12 | NULL        | NULL             | Sleep midpoint timing, rank-normalized                                             |
| 44523 | MeanSleepNumEpisodesRN                   |           12 | NULL        | NULL             | Number of sleep episodes, rank-normalized                                          |
| 44453 | MIinT2D                                  |           12 | NULL        | NULL             | Myocardial infarction in type 2 diabetes                                           |
| 44430 | mRS012v3456                              |           12 | NULL        | NULL             | Modified Rankin scale score 0-2 vs 3-6                                             |
| 44431 | mRS012v3456AdjSever                      |           12 | NULL        | NULL             | Modified Rankin scale score 0-2 vs 3-6 adj stroke severity                         |
| 44432 | mRS01v23456                              |           12 | NULL        | NULL             | Modified Rankin scale score 0-1 vs 2-6                                             |
| 44433 | mRS01v23456AdjSever                      |           12 | NULL        | NULL             | Modified Rankin scale score 0-1 vs 2-6 adj stroke severity                         |
| 44434 | mRSOrdinal                               |           12 | NULL        | NULL             | Modified Rankin scale score                                                        |
| 44435 | mRSOrdinalAdjSever                       |           12 | NULL        | NULL             | Modified Rankin scale score adj stroke severity                                    |
| 44533 | MUFA                                     |           12 | NULL        | NULL             | Total monounsaturated fatty acids                                                  |
| 44578 | NaExcretion                              |           12 | NULL        | NULL             | Urinary sodium excretion                                                           |
| 44609 | NAFLDinT2D                               |           12 | NULL        | NULL             | Non-alcoholic fatty liver disease in type 2 diabetes                               |
| 44506 | nonlobarSVS                              |           12 | NULL        | NULL             | Non-lobar intracerebral hemorrhage or small vessel ischemic stroke                 |
| 44454 | NPDRvNoDR                                |           12 | NULL        | NULL             | Non-proliferative diabetic retinopathy vs no DR                                    |
| 44455 | NPDRvNoDRwDoD                            |           12 | NULL        | NULL             | Non-proliferative diabetic retinopathy vs no DR acct DoD-GC                        |
| 44534 | PC                                       |           12 | NULL        | NULL             | Phosphocholines                                                                    |
| 44559 | PC1diet                                  |           12 | NULL        | NULL             | PC1 dietary pattern                                                                |
| 44560 | PC3diet                                  |           12 | NULL        | NULL             | PC3 dietary pattern                                                                |
| 44544 | PEAKadjBMInSI                            |           12 | NULL        | NULL             | Peak insulin response adj BMI-SI                                                   |
| 44541 | PEAKadjSI                                |           12 | NULL        | NULL             | Peak insulin response adj SI                                                       |
| 44535 | PUFA                                     |           12 | NULL        | NULL             | Total polyunsaturated fatty acids                                                  |
| 44425 | RA                                       |           12 | NULL        | NULL             | Low relative amplitude cases vs controls                                           |
| 44426 | RAQT                                     |           12 | NULL        | NULL             | Relative amplitude                                                                 |
| 44587 | RBCDistWidth                             |           12 | NULL        | NULL             | Red blood cell distribution width                                                  |
| 44558 | saltconsumption                          |           12 | NULL        | NULL             | Salt addition to food                                                              |
| 44419 | SCOOPvSTILTS                             |           12 | NULL        | NULL             | Obese vs thin                                                                      |
| 44420 | SCOOPvUKHLS                              |           12 | NULL        | NULL             | Obese vs controls                                                                  |
| 44575 | SevereC19vAll                            |           12 | NULL        | NULL             | Very severe respiratory confirmed COVID-19 vs population                           |
| 44574 | SeverevNonHospC19                        |           12 | NULL        | NULL             | Very severe respiratory confirmed vs non-hospitalized COVID-19                     |
| 44536 | SFA                                      |           12 | NULL        | NULL             | Total saturated fatty acids                                                        |
| 44500 | SleepLong                                |           12 | NULL        | NULL             | Long sleep duration                                                                |
| 44501 | SleepShort                               |           12 | NULL        | NULL             | Short sleep duration                                                               |
| 44570 | smeq_binary                              |           12 | NULL        | NULL             | Chronotype (binary sMEQ score)                                                     |
| 44571 | smeq_cont                                |           12 | NULL        | NULL             | Chronotype (continuous sMEQ score)                                                 |
| 44399 | SmokingFGint                             |           12 | NULL        | NULL             | Smoking-fasting glucose interaction                                                |
| 44400 | SmokingFGjoint                           |           12 | NULL        | NULL             | Smoking-fasting glucose interaction, joint 2-degree-of-freedom test                |
| 44401 | SmokingFGmain                            |           12 | NULL        | NULL             | Smoking-fasting glucose interaction, main effect model                             |
| 44402 | SmokingT2Dint                            |           12 | NULL        | NULL             | Smoking-T2D interaction                                                            |
| 44403 | SmokingT2Djoint                          |           12 | NULL        | NULL             | Smoking-T2D interaction, joint 2-degree-of-freedom test                            |
| 44404 | SmokingT2Dmain                           |           12 | NULL        | NULL             | Smoking-T2D interaction, main effect model                                         |
| 44428 | SnoringAdjBMI                            |           12 | NULL        | NULL             | Snoring adj BMI                                                                    |
| 44517 | SpO2avg                                  |           12 | NULL        | NULL             | Average oxyhemoglobin saturation during sleep                                      |
| 44515 | SpO2min                                  |           12 | NULL        | NULL             | Minimum oxyhemoglobin saturation during sleep                                      |
| 44516 | SpO2per90                                |           12 | NULL        | NULL             | Percentage of sleep with oxyhemoglobin saturation under 90%                        |
| 44524 | StDevSleepDurationRN                     |           12 | NULL        | NULL             | Standard deviation of sleep duration, rank-normalized                              |
| 44421 | STILTSvUKHLS                             |           12 | NULL        | NULL             | Controls vs thin                                                                   |
| 44456 | StrokeinT2D                              |           12 | NULL        | NULL             | Stroke in type 2 diabetes                                                          |
| 44457 | SVI                                      |           12 | NULL        | NULL             | Stroke volume (BSA-indexed)                                                        |
| 44507 | T2DadjBMI                                |           12 | NULL        | NULL             | Type 2 diabetes adj BMI                                                            |
| 44566 | TBLH-BMD                                 |           12 | NULL        | NULL             | Total body (less head) bone mineral density                                        |
| 44424 | TFR                                      |           12 | NULL        | NULL             | you                                                                    |
| 44508 | toastCE                                  |           12 | NULL        | NULL             | TOAST cardio-aortic embolism                                                       |
| 44509 | toastDETER                               |           12 | NULL        | NULL             | TOAST other determined                                                             |
| 44510 | toastLAA                                 |           12 | NULL        | NULL             | TOAST large artery atherosclerosis                                                 |
| 44511 | toastSAO                                 |           12 | NULL        | NULL             | TOAST small artery occlusion                                                       |
| 44512 | toastUNDETER                             |           12 | NULL        | NULL             | TOAST other undetermined                                                           |
| 44581 | UACR_DM                                  |           12 | NULL        | NULL             | Urinary albumin-to-creatinine ratio in subjects with diabetes                      |
| 44513 | Urate                                    |           12 | NULL        | NULL             | Serum urate                                                                        |
| 44554 | WCadjBMISMK                              |           12 | NULL        | NULL             | Waist circumference adj BMI-smoking status                                         |
| 44555 | WHRadjBMISMK                             |           12 | NULL        | NULL             | Waist-hip ratio adj BMI-smoking status                                             |

-- done

