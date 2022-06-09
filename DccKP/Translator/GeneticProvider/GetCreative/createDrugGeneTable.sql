


drop table if exists tran_creative.drug_gene;
create table tran_creative.drug_gene (
  id                        int not null auto_increment primary key,
  drug_id                   varchar(250) not null,                        
  drug_name                 varchar(2000) not null,                        
  gene_id                   varchar(250) not null,                        
  predicate                 varchar(250) not null,                        
  last_updated              timestamp default CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);



