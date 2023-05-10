
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


drop table if exists pubmed_gpt.pmd_keyword;
create table pubmed_gpt.pmd_keyword (
  id                        int not null auto_increment primary key,
  keyword                   varchar(400),
  pubmed_curie              varchar(50),
  translator_curie          varchar(50),
  translator_type           varchar(50),
  searched_on_translator    enum('Y', 'N') default 'N',
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
);


drop table if exists pubmed_gpt.pmd_link_keyword_abstract;
create table pubmed_gpt.pmd_link_keyword_abstract (
  id                        int not null auto_increment primary key,
  keyword_id                int(9) not null,
  abstract_id               int(9) not null,
  offset                    int(9),
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
);
alter table pubmed_gpt.pmd_link_keyword_abstract add index pub_lnk_key_idx (keyword_id);
alter table pubmed_gpt.pmd_link_keyword_abstract add index pub_lnk_abs_idx (abstract_id);



-- DEPRECATED
-- gene table
drop table if exists pubmed_gpt.pmd_gene;
create table pubmed_gpt.pmd_gene (
  id                        int not null auto_increment primary key,
  gene_name                 varchar(400),
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
);


-- gene/abstract link table 
drop table if exists pubmed_gpt.pmd_link_gene_abstract;
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



-- adding curie, type for keywords 
update pubmed_gpt.pmd_keyword keyw
join tran_test_202303.comb_node_ontology node on node.node_name collate utf8mb4_unicode_ci = keyw.keyword
join tran_test_202303.comb_lookup_type look on look.type_id = node.node_type_id
where keyw.translator_curie is null
set keyw.translator_curie = node.ontology_id, keyw.translator_type = look.type_name;



-- queries
select count(id), paper_date from pubmed_gpt.pmd_abstract group by paper_date order by paper_date desc;

select count(id), year(paper_date), month(paper_date) from pubmed_gpt.pmd_abstract 
group by year(paper_date), month(paper_date) order by year(paper_date) desc, month(paper_date) desc;

select count(id) from pmd_abstract;

select count(id) from pmd_abstract where lower(journal_title) like ('%gene%');


select distinct journal_title from pmd_abstract where lower(journal_title) like ('%genet%') order by journal_title;

select abstract.pubmed_id, abstract.paper_title, link.offset, keyword.keyword
from pmd_abstract abstract, pmd_link_keyword_abstract link, pmd_keyword keyword
where abstract.id = link.abstract_id and link.keyword_id = keyword.id
limit 50;

select count(id) from pmd_keyword;

select * from keyword where translator_curie is not null
order by keyword;

-- join keywords with translator nodes case insensitive
select keyword.keyword, node.ontology_id, node.id, look.type_name
from pmd_keyword keyword, tran_test_202303.comb_node_ontology node , tran_test_202303.comb_lookup_type look
where lower(node.node_name) collate utf8mb4_unicode_ci = lower(keyword.keyword) and look.type_id = node.node_type_id
order by keyword.keyword;


-- join keywords with translator nodes case used 
select keyword.keyword, node.ontology_id, node.id, look.type_name
from pmd_keyword keyword, tran_test_202303.comb_node_ontology node , tran_test_202303.comb_lookup_type look
where node.node_name collate utf8mb4_unicode_ci = keyword.keyword and look.type_id = node.node_type_id
order by keyword.keyword;

-- count of keywords by paper 
select abs.pubmed_id, count(keyw.id) as count
from pmd_abstract abs, pmd_keyword keyw, pmd_link_keyword_abstract link
where abs.id = link.abstract_id and link.keyword_id = keyw.id
group by abs.pubmed_id
order by count, abs.pubmed_id;

-- count of keywords with curies by paper 
select abs.pubmed_id, count(keyw.id) as count
from pmd_abstract abs, pmd_keyword keyw, pmd_link_keyword_abstract link
where abs.id = link.abstract_id and link.keyword_id = keyw.id and keyw.translator_curie is not null
group by abs.pubmed_id
order by count, abs.pubmed_id;

-- count of keywords with curies by paper, at leat 3 keywords
select abs.pubmed_id, count(keyw.id) as count
from pmd_abstract abs, pmd_keyword keyw, pmd_link_keyword_abstract link
where abs.id = link.abstract_id and link.keyword_id = keyw.id 
and keyw.translator_curie is not null and link.offset > 0
group by abs.pubmed_id
order by count, abs.pubmed_id;

select abs.pubmed_id, keyw.keyword, keyw.translator_curie, keyw.translator_type, link.offset
from pmd_abstract abs, pmd_keyword keyw, pmd_link_keyword_abstract link 
where abs.id = link.abstract_id and link.keyword_id = keyw.id 
and keyw.translator_curie is not null and link.offset > 0
order by abs.pubmed_id;


