
-- abstracts table 
drop table if exists pubmed_gpt.pmd_abstract;
create table pubmed_gpt.pmd_abstract (
  id                        int not null auto_increment primary key,
  pubmed_id                 int(10),
  paper_date                date,
  paper_title               varchar(4000),
  journal_title             varchar(1000),
  abstract_text             varchar(10000),
  processed                 enum('Y', 'N'),
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
);


-- gene table
drop table if exists pubmed_gpt.pubmed_gpt.pmd_gene;
create table pubmed_gpt.pmd_gene (
  id                        int not null auto_increment primary key,
  gene_name                 varchar(400),
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
);


-- gene/abstract link table 
drop table if exists pubmed_gpt.pubmed_gpt.pmd_link_gene_abstract;
create table pubmed_gpt.pmd_link_gene_abstract (
  id                        int not null auto_increment primary key,
  gene_id                   int(9) not null,
  abstract_id               int(9) not null,
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
);
alter table pubmed_gpt.pmd_link_gene_abstract add index pub_lnk_gen_idx (gene_id);
alter table pubmed_gpt.pmd_link_gene_abstract add index pub_lnk_abs_idx (abstract_id);



-- populate the gene table from the translator schema
insert into pubmed_gpt.pmd_gene (gene_name)
select distinct node_code from tran_test_202303.comb_node_ontology where node_type_id = 2 order by node_code;




-- queries
select distinct paper_date from pubmed_gpt.pmd_abstract order by paper_date;



