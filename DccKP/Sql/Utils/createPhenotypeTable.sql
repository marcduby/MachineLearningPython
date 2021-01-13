

drop table if exists phenotype_lookup;
create table phenotype_lookup (
  id                        int not null auto_increment primary key,
  phenotype_code            varchar(1000) not null,
  phenotype                 varchar(1000),
  dichotomous               varchar(10),
  category                  varchar(100),
  group_name                varchar(100),
  mondo_id                  varchar(100),
  tran_mondo_id             varchar(100),
  mondo_name                varchar(200),
  efo_id                    varchar(100),
  tran_efo_id               varchar(100),
  efo_name                  varchar(200)
  tran_lookup_id            varchar(100),
  tran_lookup_name          varchar(200),
);

desc phenotype_lookup;

alter table phenotype_lookup add index plook_tran_efo_id_idx (tran_efo_id);
alter table phenotype_lookup add index plook_tran_mondo_id_idx (tran_mondo_id);
alter table phenotype_lookup add index plook_tran_lookup_id_idx (tran_lookup_id);



