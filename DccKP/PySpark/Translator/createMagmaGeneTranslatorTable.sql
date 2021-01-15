


drop table if exists magma_gene_phenotype;
create table magma_gene_phenotype (
  phenotype_code            varchar(100) not null,
  phenotype_ontology_id     varchar(100),
  phenotype                 varchar(1000),
  dichotomous               varchar(10),
  biolink_category          varchar(100),
  group_name                varchar(100),
  ncbi_id                   varchar(100),
  gene                      varchar(50),
  p_value                   double,
  id                        int not null auto_increment primary key
);

desc magma_gene_phenotype;

-- add indices
alter table magma_gene_phenotype add index mgp_phenotype_id_idx (phenotype_ontology_id);
alter table magma_gene_phenotype add index mgp_ncbi_id_idx (ncbi_id);
alter table magma_gene_phenotype add index mgp_p_value_idx (p_value);
alter table magma_gene_phenotype add index mgp_category_idx (biolink_category);


-- load the data to mysql
-- cp results/part-00000-9f731952-6cc0-4651-bf58-a57d73c672b8-c000.csv magma_gene_phenotype.tsv
-- mysqlimport --ignore-lines=1 --fields-terminated-by='\t' --local -u root -p tran_genepro magma_gene_phenotype.tsv

