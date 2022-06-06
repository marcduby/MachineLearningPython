


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



drop table if exists tran_upkeep.data_pathway_genes;
create table tran_upkeep.data_pathway_genes (
  id                           int not null auto_increment primary key,
  pathway_id                   int(9) not null,
  gene_code                   varchar(200) not null,
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
);

alter table tran_upkeep.data_pathway_genes add index path_gen_path_id_idx (pathway_id);




-- queries
select *
from data_pathway pathway
where pathway.pathway_code = 'GO:0045444';


select *
from data_pathway pathway, data_pathway_genes gene
where gene.pathway_id = pathway.id 
and pathway.pathway_code = 'GO:0045444';

select *
from data_pathway pathway, data_pathway_genes gene
where gene.pathway_id = pathway.id 
and pathway.pathway_code = 'GO:0050872';

select pathway.pathway_code, gene.*
from data_pathway pathway, data_pathway_genes gene
where gene.pathway_id = pathway.id 
and pathway.pathway_code in ('GO:0009256', 'GO:0042398');



-- update node ontology table pathway data name based on loaded pathway data
select node.node_code, node_name, pathway.pathway_code, pathway.pathway_updated_name
from comb_node_ontology node, tran_upkeep.data_pathway pathway
where node.node_code COLLATE utf8mb4_general_ci = pathway.pathway_code;


update comb_node_ontology node
join tran_upkeep.data_pathway pathway on node.node_code COLLATE utf8mb4_general_ci = pathway.pathway_code
set node.node_name = pathway.pathway_updated_name
where node.node_type_id = 4;

 and  node.node_code like '%3';
