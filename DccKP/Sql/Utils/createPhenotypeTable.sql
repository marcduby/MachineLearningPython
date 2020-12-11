

drop table if exists phenotype_lookup;
create table phenotype_lookup (
  id                        int not null auto_increment primary key,
  phenotype_code            varchar(1000) not null,
  phenotype                 varchar(1000),
  dichotomous               varchar(10),
  category                  varchar(100),
  group_name                varchar(100),
  mondo_id                  varchar(100),
  efo_id                    varchar(100),
  tran_efo_id               varchar(100)
);

desc phenotype_lookup;



