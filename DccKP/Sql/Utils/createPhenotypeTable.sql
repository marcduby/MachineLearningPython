

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
  efo_name                  varchar(200),
  tran_lookup_id            varchar(100),
  tran_lookup_name          varchar(200)
);

desc phenotype_lookup;

alter table phenotype_lookup add index plook_tran_efo_id_idx (tran_efo_id);
alter table phenotype_lookup add index plook_tran_mondo_id_idx (tran_mondo_id);
alter table phenotype_lookup add index plook_tran_lookup_id_idx (tran_lookup_id);





drop table if exists phenotype_id_lookup;
create table phenotype_id_lookup (
  id                        int not null auto_increment primary key,
  phenotype_code            varchar(100) not null,
  ontology_name             varchar(100),
  dichotomous               varchar(10),
  category                  varchar(100),
  group_name                varchar(100),
  tran_lookup_id            varchar(100),
  tran_lookup_name          varchar(200)
);

desc phenotype_id_lookup;

alter table phenotype_id_lookup add index p2look_tran_code_idx (phenotype_code);
alter table phenotype_id_lookup add index p2look_tran_lookup_id_idx (tran_lookup_id);



-- 20210115 - 220 phenotypes on 01/15/21 before fix to avoid dup names
-- 20210115 - 210 after dup names taken out and reloaded

