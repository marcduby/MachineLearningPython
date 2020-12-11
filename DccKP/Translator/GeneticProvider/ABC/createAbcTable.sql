

drop table if exists abc_gene_phenotype;
create table abc_gene_phenotype (
  id                        int not null auto_increment primary key,
  gene                      varchar(100) not null,
  gene_ncbi_id              varchar(100),
  phenotype                 varchar(100),
  phenotype_efo_id          varchar(100),
  category                  varchar(100),
  p_value                   double
);
alter table abc_gene_phenotype add edge_id varchar(100);

-- add the edge id
update abc_gene_phenotype set edge_id = concat('ABC_GENE_', cast(id as char));

-- indices
alter table abc_gene_phenotype add index abc_gene_idx (gene);
alter table abc_gene_phenotype add index abc_phenotype_idx (phenotype);
alter table abc_gene_phenotype add index abc_gene_id_idx (gene_ncbi_id);
alter table abc_gene_phenotype add index abc_phenotype_id_idx (phenotype_efo_id);

-- update the gene ids
update abc_gene_phenotype abc join gene_lookup look 
on abc.gene = look.gene
set abc.gene_ncbi_id = look.ncbi_id;
select * from abc_gene_phenotype limit 20;

-- update the phenotype ids
update abc_gene_phenotype abc join phenotype_lookup look 
on abc.phenotype = look.phenotype_code
set abc.phenotype_efo_id = look.tran_efo_id, abc.category = look.category;
select * from abc_gene_phenotype limit 20;

-- update the phenotype category
update abc_gene_phenotype abc join category_lookup cat 
on abc.phenotype_efo_id = cat.disease
set abc.category = cat.category;
select * from abc_gene_phenotype limit 20;



