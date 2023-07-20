
-- search table
drop table if exists pgpt_search;
create table pgpt_search (
  id                        int not null auto_increment primary key,
  name                      varchar(500) not null,
  terms                     varchar(5000) not null,
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
);
-- indices
alter table comb_node_ontology add index node_ont_node_cde_idx (node_code);
alter table comb_node_ontology add index node_ont_node_typ_idx (node_type_id);
alter table comb_node_ontology add index node_ont_ont_idx (ontology_id);


-- keywords tables
drop table if exists pgpt_keyword;
create table pgpt_keyword (
  id                        int not null auto_increment primary key,
  keyword                   varchar(5000) not null,
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
);

-- search/keyword link table
drop table if exists pgpt_search_keyword;
create table pgpt_search_keyword (
  id                        int not null auto_increment primary key,
  search_id                 int(9) not null,
  keyword_id                int(9) not null,
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
);
alter table pgpt_search_keyword add index pgpt_ser_key_ser (search_id);
alter table pgpt_search_keyword add index pgpt_ser_key_key (keyword_id);


-- paper table
drop table if exists pgpt_paper;
create table pgpt_paper (
  pubmed_id                 int not null primary key,
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
);


-- paper/search table
drop table if exists pgpt_search_paper;
create table pgpt_search_paper (
  id                        int not null auto_increment primary key,
  search_id                 int(9) not null,
  paper_id                  int(9) not null,
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
);
alter table pgpt_search_paper add index pgpt_ser_pap_ser (search_id);
alter table pgpt_search_paper add index pgpt_ser_pap_pap (paper_id);


-- download paper table
drop table if exists pgpt_paper_abstract;
create table pgpt_paper_abstract (
  id                        int not null auto_increment primary key,
  pubmed_id                 int(9) null,
  title                     varchar(1000) not null,
  abstract                  varchar(4000) not null,
  journal_name              varchar(2000) not null,
  document_level            int(9) not null,
  paper_year                int(9) null,
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
);
alter table pgpt_paper_abstract add index pgpt_pap_abs_pub (pubmed_id);



drop table if exists pgpt_gpt_paper;
create table pgpt_gpt_paper (
  id                        int not null auto_increment primary key,
  parent_id                 int(9) not null,
  child_id                  int(9) not null,
  search_id                 int(9) not null,
  document_level            int(9) not null,
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
);
alter table pgpt_gpt_paper add index pgpt_gpt_pap_par (parent_id);
alter table pgpt_gpt_paper add index pgpt_gpt_pap_chi (child_id);
alter table pgpt_gpt_paper add index pgpt_gpt_pap_sea (search_id);


-- start data
insert into pgpt_keyword (keyword) values ('PPARG');
insert into pgpt_keyword (keyword) values ('human');

insert into pgpt_search (name, terms) values('pparg search', 'PPARG,human');

insert into pgpt_search_keyword (search_id, keyword_id) values(1, 1);
insert into pgpt_search_keyword (search_id, keyword_id) values(1, 2);


-- query
select link.search_id, se.name, se.terms, link.keyword_id, k.keyword
from pgpt_search se, pgpt_keyword k, pgpt_search_keyword link
where se.id = link.search_id and k.id = link.keyword_id
order by k.keyword;

select distinct journal_name from pgpt_paper_abstract order by journal_name;

select * from pgpt_gpt_paper order by child_id;

select id, document_level, abstract from pgpt_paper_abstract where pubmed_id is null order by document_level, id;

select id, abstract from pgpt_paper_abstract 
where document_level = 1 and id not in (select child_id from pgpt_gpt_paper where search_id = 1) limit 5;


select gpt.parent_id, gpt.child_id, gpt.document_level 
from pgpt_gpt_paper gpt, pgpt_paper_abstract paper 
where gpt.child_id = paper.id
order by gpt.child_id;


-- cleanup
delete from pgpt_paper_abstract where pubmed_id is null and id not in (select distinct parent_id from pgpt_gpt_paper);

delete from pgpt_gpt_paper;
delete from pgpt_paper_abstract where pubmed_id is null and id not in (select distinct parent_id from pgpt_gpt_paper);

update pgpt_gpt_paper node
  join pgpt_paper_abstract paper on node.child_id = paper.pubmed_id
  set node.child_id = paper.id;









