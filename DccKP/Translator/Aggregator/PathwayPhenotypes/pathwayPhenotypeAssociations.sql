

-- create table for loading pathway name to id data
drop table if exists tran_upkeep.agg_pathway_phenotype;
create table tran_upkeep.agg_pathway_phenotype (
  id                           int not null auto_increment primary key,
  pathway_code                 varchar(250) not null,
  phenotype_code               varchar(50) not null,
  number_genes                 int(9) not null,
  beta                         double not null,
  beta_standard_error          double not null,
  standard_error               double not null,
  p_value                      double not null,
  date_created                 datetime DEFAULT CURRENT_TIMESTAMP
);




alter table tran_upkeep.agg_pathway_phenotype add index path_phen_phen_cde_idx (phenotype_code);



-- update node ontolgy pathway rowsbased on download names




-- queries 
select count(id), phenotype_code from tran_upkeep.agg_pathway_phenotype group by phenotype_code;


