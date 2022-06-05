


-- create the data
drop table if exists tran_upkeep.data_pathway;
create table tran_upkeep.data_pathway (
  id                        int not null auto_increment primary key,
  pathway_code              varchar(250) not null,                        
  pathway_name              varchar(2000) not null,                        
  pathway_updated_name      varchar(2000) not null,             
  gene_count                int(9) not null,           
  last_updated              timestamp default CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);


-- create indexes
create index dt_pathway_info_cde on tran_upkeep.data_pathway(pathway_code);
