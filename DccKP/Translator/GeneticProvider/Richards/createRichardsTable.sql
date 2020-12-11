
create table gene_phenotype (
  id                        int not null auto_increment primary key,
  gene                      varchar(100) not null,
  phenotype                 varchar(10),
  prob                      double
);


alter table gene_phenotype add phenotype_efo varchar(50);
alter table gene_phenotype add gene_ncbi varchar(50);
alter table gene_phenotype add index gene_idx (gene)

-- data 
update gene_phenotype set phenotype_efo = 'EFO_0004570' where phenotype = 'dbilirubin';

update gene_phenotype set phenotype_efo = 'EFO:0004611' where phenotype = 'ldl';
update gene_phenotype set phenotype_efo = 'EFO:0009270' where phenotype = 'ebmd';
update gene_phenotype set phenotype_efo = 'EFO:0004468' where phenotype = 'glucose';
update gene_phenotype set phenotype_efo = 'EFO:0006336' where phenotype = 'dbp';
update gene_phenotype set phenotype_efo = 'EFO:0004838' where phenotype = 'calcium';
update gene_phenotype set phenotype_efo = 'EFO:0001360' where phenotype = 't2d';
update gene_phenotype set phenotype_efo = 'EFO:0006335' where phenotype = 'sbp';
update gene_phenotype set phenotype_efo = 'EFO:0004530' where phenotype = 'tg';
update gene_phenotype set phenotype_efo = 'EFO:0004339' where phenotype = 'height';
update gene_phenotype set phenotype_efo = 'EFO:0004305' where phenotype = 'rbc';
update gene_phenotype set phenotype_efo = 'EFO:0004705' where phenotype = 'lowtsh';


drop table if exists gene_ncbi;
create table gene_ncbi (
  id                        int not null auto_increment primary key,
  gene                      varchar(100) not null,
  ncbi_id                   varchar(100),
  ncbi_id_int               int(9)
);
alter table gene_ncbi add index gene_ncbi_idx (gene)


UPDATE gene_phenotype phe JOIN gene_ncbi ncbi ON (phe.gene = ncbi.gene)
SET phe.gene_ncbi = ncbi.ncbi_id;

select count(*) from gene_phenotype where gene_ncbi is null;

UPDATE gene_phenotype phe JOIN gene_ncbi_small ncbi ON (phe.gene = ncbi.gene)
SET phe.gene_ncbi = ncbi.ncbi_id where phe.gene_ncbi is null;


update gene_phenotype set gene = 'DEC1' where gene = '1-Dec';
update gene_phenotype set gene = 'MAR1' where gene = '1-Mar';
update gene_phenotype set gene = 'MAR2' where gene = '2-Mar';
update gene_phenotype set gene = 'MAR9' where gene = '9-Mar';
update gene_phenotype set gene = 'MAR10' where gene = '10-Mar';
update gene_phenotype set gene = 'SEP1' where gene = '1-Sep';
update gene_phenotype set gene = 'SEP2' where gene = '2-Sep';
update gene_phenotype set gene = 'SEP3' where gene = '3-Sep';
update gene_phenotype set gene = 'SEP4' where gene = '4-Sep';
update gene_phenotype set gene = 'SEP5' where gene = '5-Sep';
update gene_phenotype set gene = 'SEP7' where gene = '7-Sep';
update gene_phenotype set gene = 'SEP8' where gene = '8-Sep';
update gene_phenotype set gene = 'SEP9' where gene = '9-Sep';
update gene_phenotype set gene = '' where gene = '';
update gene_phenotype set gene = '' where gene = '';
update gene_phenotype set gene = '' where gene = '';
update gene_phenotype set gene = '' where gene = '';



create table gene_ncbi_small as 
select min(ncbi_id) as gene_id, gene from gene_ncbi group by gene;

select * from gene_phenotype phe, gene_ncbi ncbi where phe.gene = ncbi.gene limit 10;

create table gene_ncbi_load (
  id                        int not null auto_increment primary key,
  gene                      varchar(100) not null,
  ncbi_id                   int(9)
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




create table RICHARDS_GENES (
  ID            varchar(255),
  GENE          varchar(255),
  DISEASE       varchar(255),
  CATEGORY      varchar(255),
  PROBABILITY   double
)

drop table if exists richards_gene;
create table richards_gene as
select concat('RC_GENES_', cast(id as char)) as id, gene_ncbi as gene, gene as gene_name, phenotype_efo as phenotype,
  phenotype as phenotype_name, prob as probability from richards_gene.gene_phenotype;

alter table richards_gene add category varchar(100);
alter table richards_gene add index rc_gene_idx (gene);
alter table richards_gene add index rc_phenotype_idx (phenotype);

update richards_gene rc join category_lookup cat on rc.phenotype = cat.disease
set rc.category = cat.category;


update richards_gene set phenotype = 'EFO:0004611' where phenotype_name = 'ldl';
update richards_gene set phenotype = 'EFO:0009270' where phenotype_name = 'ebmd';
update richards_gene set phenotype = 'EFO:0004468' where phenotype_name = 'glucose';
update richards_gene set phenotype = 'EFO:0006336' where phenotype_name = 'dbp';
update richards_gene set phenotype = 'EFO:0004838' where phenotype_name = 'calcium';
update richards_gene set phenotype = 'EFO:0001360' where phenotype_name = 't2d';
update richards_gene set phenotype = 'EFO:0006335' where phenotype_name = 'sbp';
update richards_gene set phenotype = 'EFO:0004530' where phenotype_name = 'tg';
update richards_gene set phenotype = 'EFO:0004339' where phenotype_name = 'height';
update richards_gene set phenotype = 'EFO:0004305' where phenotype_name = 'rbc';
update richards_gene set phenotype = 'EFO:0004705' where phenotype_name = 'lowtsh';

select distinct rc.phenotype from richards_gene rc, category_lookup cat where cat.disease = rc.phenotype;

create table category_lookup as 
select distinct category, disease from MAGMA_GENES;
