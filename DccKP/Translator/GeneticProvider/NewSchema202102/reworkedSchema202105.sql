

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







