
drop table if exists clingen_gene_phenotype;
create table clingen_gene_phenotype (
  id                        int not null auto_increment primary key,
  gene                      varchar(100) not null,
  gene_id                   varchar(100),
  phenotype                 varchar(1000),
  phenotype_id              varchar(50),
  provenance                varchar(50),
  classification            varchar(100)
);


