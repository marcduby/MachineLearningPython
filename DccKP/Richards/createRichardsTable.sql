
create table gene_phenotype (
  id                        int not null auto_increment primary key,
  gene                      varchar(100) not null,
  phenotype                 varchar(10),
  prob                      double
);


alter table gene_phenotype add phenotype_efo varchar(50);
alter table gene_phenotype add gene_ncbi varchar(50);


-- data 
update gene_phenotype set phenotype_efo = 'EFO_0004570' where phenotype = 'dbilirubin';

update gene_phenotype set phenotype_efo = 'EFO_0004611' where phenotype = 'ldl';
update gene_phenotype set phenotype_efo = 'EFO_0009270' where phenotype = 'ebmd';
update gene_phenotype set phenotype_efo = 'EFO_0004468' where phenotype = 'glucose';
update gene_phenotype set phenotype_efo = 'EFO_0006336' where phenotype = 'dbp';
update gene_phenotype set phenotype_efo = 'EFO_0004838' where phenotype = 'calcium';
update gene_phenotype set phenotype_efo = 'EFO_0001360' where phenotype = 't2d';
update gene_phenotype set phenotype_efo = 'EFO_0006335' where phenotype = 'sbp';
update gene_phenotype set phenotype_efo = 'EFO_0004530' where phenotype = 'tg';
update gene_phenotype set phenotype_efo = 'EFO_0004339' where phenotype = 'height';
update gene_phenotype set phenotype_efo = 'EFO_0004305' where phenotype = 'rbc';
update gene_phenotype set phenotype_efo = 'EFO_0004705' where phenotype = 'lowtsh';



create table gene_ncbi (
  id                        int not null auto_increment primary key,
  gene                      varchar(100) not null,
  ncbi_id                   varchar(100)
);



-- mysql> select distinct phenotype, phenotype_efo from gene_phenotype;
-- +------------+---------------+
-- | phenotype  | phenotype_efo |
-- +------------+---------------+
-- | dbilirubin | NULL          |
-- | ldl        | NULL          |
-- | ebmd       | NULL          |
-- | glucose    | NULL          |
-- | dbp        | NULL          |
-- | lowtsh     | NULL          |
-- | calcium    | NULL          |
-- | rbc        | NULL          |
-- | t2d        | NULL          |
-- | sbp        | NULL          |
-- | tg         | NULL          |
-- | height     | NULL          |
-- +------------+---------------+
-- 12 rows in set (0.04 sec)

-- mysql> 

