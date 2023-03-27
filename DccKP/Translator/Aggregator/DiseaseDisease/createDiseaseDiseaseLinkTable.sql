
drop table if exists tran_upkeep.agg_disease_correlation;
create table tran_upkeep.agg_disease_correlation (
  id                        int not null auto_increment primary key,
  phenotype_id              varchar(100) not null,
  other_phenotype_id        varchar(100) not null,
  p_value                   double not null,
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
  last_updated              timestamp default CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

alter table agg_aggregator_phenotype add unique index u_phenotype_id_idx (phenotype_id);




