

-- create the data
drop table if exists data_pathway_similarity;
create table data_pathway_similarity (
  id                        int not null auto_increment primary key,
  subject_pathway_code      varchar(250) not null,                        
  object_pathway_code       varchar(250) not null,                        
  google_distance           double,
  google_distance_min       double,
  rank_percentage_goo       double,
  rank_percentage_goo_min   double,
  last_updated              timestamp default CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);


-- create indexes
create index dt_pathway_sim_subj on data_pathway_similarity(subject_pathway_code);
create index dt_pathway_sim_obj on data_pathway_similarity(object_pathway_code);
create index dt_pathway_sim_goo_min_idx on data_pathway_similarity(google_distance_min);
create index dt_pathway_sim_goo_idx on data_pathway_similarity(google_distance);




-- STAGING table - will load data only in one direction
-- create the data
drop table if exists load_pathway_similarity;
create table load_pathway_similarity (
  id                        int not null auto_increment primary key,
  subject_pathway_code      varchar(250) not null,                        
  object_pathway_code       varchar(250) not null,                        
  google_distance           double,
  google_distance_min       double,
  rank_percentage_goo       double,
  rank_percentage_goo_min   double,
  last_updated              timestamp default CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

create index dt_pathway_ld_sim_goo_min_idx on load_pathway_similarity(google_distance_min);
create index dt_pathway_ld_sim_goo_idx on load_pathway_similarity(google_distance);





-- scratch
select * from data_pathway_similarity where object_pathway_code = 'R-HSA-381340' order by google_distance desc;


select go.subject_pathway_code, sub.gene_count, go.object_pathway_code, ob.gene_count, go.google_distance, sub.pathway_updated_name 
from data_pathway_similarity go, data_pathway sub, data_pathway ob
where go.object_pathway_code = 'R-HSA-381340' and go.subject_pathway_code = sub.pathway_code and go.object_pathway_code = ob.pathway_code
order by go.google_distance desc
limit 20;



select go.subject_pathway_code as subj_pathway, sub.gene_count, go.object_pathway_code as obj_pathway, 
  go.google_distance as jaccard, go.google_distance_min as intercept_min, 
  sub.pathway_updated_name as subj_pathway_name
from data_pathway_similarity go, data_pathway sub
where go.object_pathway_code = 'R-HSA-381340' and go.subject_pathway_code = sub.pathway_code
and go.subject_pathway_code in ('GO:0048869', 'GO:0030154', 'GO:45444', 'GO:0050872', 'GO:0050873', 'GO:0045598', 'GO:0045599', 'GO:0045600')
order by go.google_distance_min desc
limit 20;

-- percent rank
drop table if exists calc_pathway_similarity;
create table calc_pathway_similarity as
select *,
PERCENT_RANK() over (
order by google_distance desc
) jaccard_percent_rank
from load_pathway_similarity;

create index clc_pathway_sim_subj on calc_pathway_similarity(subject_pathway_code);
create index clc_pathway_sim_obj on calc_pathway_similarity(object_pathway_code);

update data_pathway_similarity data
join calc_pathway_similarity calc on data.subject_pathway_code = calc.subject_pathway_code
and data.object_pathway_code = calc.object_pathway_code
set data.rank_percentage_goo = calc.jaccard_percent_rank;

update data_pathway_similarity data
join calc_pathway_similarity calc on data.object_pathway_code = calc.subject_pathway_code
  and data.subject_pathway_code = calc.object_pathway_code
set data.rank_percentage_goo = calc.jaccard_percent_rank;


drop table if exists calc_pathway_similarity_min;
create table calc_pathway_similarity_min as
select *,
PERCENT_RANK() over (
order by google_distance_min desc
) intercept_percent_rank
from load_pathway_similarity;

create index clc_pathway_sim_subj on calc_pathway_similarity(subject_pathway_code);
create index clc_pathway_sim_obj on calc_pathway_similarity(object_pathway_code);

update data_pathway_similarity data
join calc_pathway_similarity_min calc on data.subject_pathway_code = calc.subject_pathway_code
and data.object_pathway_code = calc.object_pathway_code
set data.rank_percentage_goo_min = calc.intercept_percent_rank;

update data_pathway_similarity data
join calc_pathway_similarity_min calc on data.object_pathway_code = calc.subject_pathway_code
  and data.subject_pathway_code = calc.object_pathway_code
set data.rank_percentage_goo_min = calc.intercept_percent_rank;



drop table if exists calc_pathway_similarity;
create table calc_pathway_similarity as
select *,
RANK() over (
order by google_distance desc
) google_distance_rank
from load_pathway_similarity;


limit 20;

