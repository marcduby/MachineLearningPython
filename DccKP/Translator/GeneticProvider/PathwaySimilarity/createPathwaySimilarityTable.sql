

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


-- scratch
select * from data_pathway_similarity where object_pathway_code = 'R-HSA-381340' order by google_distance desc;


select go.subject_pathway_code, sub.gene_count, go.object_pathway_code, ob.gene_count, go.google_distance, sub.pathway_updated_name 
from data_pathway_similarity go, data_pathway sub, data_pathway ob
where go.object_pathway_code = 'R-HSA-381340' and go.subject_pathway_code = sub.pathway_code and go.object_pathway_code = ob.pathway_code
order by go.google_distance desc
limit 20;



select go.subject_pathway_code, sub.gene_count, go.object_pathway_code, go.google_distance, sub.pathway_updated_name 
from data_pathway_similarity go, data_pathway sub
where go.object_pathway_code = 'R-HSA-381340' and go.subject_pathway_code = sub.pathway_code
and go.subject_pathway_code in ('GO:0048869', 'GO:0030154', 'GO:45444', 'GO:0050872', 'GO:0050873', 'GO:0045598', 'GO:0045599', 'GO:0045600')
order by go.google_distance desc
limit 20;


