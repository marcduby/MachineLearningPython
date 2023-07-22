
-- search table
drop table if exists pgpt_search;
create table pgpt_search (
  id                        int not null auto_increment primary key,
  name                      varchar(500) not null,
  terms                     varchar(5000) null,
  gene                      varchar(100) null,
  ready                     enum('Y', 'N') default 'N' not null,
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
);

-- alter table pgpt_search add column ready enum('Y', 'N') default 'N' not null;

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
  search_top_level_of       int(9) null,
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
);
alter table pgpt_paper_abstract add index pgpt_pap_abs_pub (pubmed_id);

alter table pgpt_paper_abstract add column search_top_level_of int(9) null;

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
insert into pgpt_keyword (keyword) values ('UBE2NL');
insert into pgpt_keyword (keyword) values ('LDLR');
insert into pgpt_keyword (keyword) values ('MAP3K15');
insert into pgpt_keyword (keyword) values ('INHBE');
insert into pgpt_keyword (keyword) values ('GIGYF1');
insert into pgpt_keyword (keyword) values ('GPR75');

insert into pgpt_search (name, terms, gene) values('pparg search', 'PPARG,human', 'PPARG');
insert into pgpt_search (name, terms, gene) values('UBE2NL search', 'UBE2NL,human', 'UBE2NL');
insert into pgpt_search (name, terms, gene) values('LDLR search', 'LDLR,human', 'LDLR');
insert into pgpt_search (name, terms, gene) values('MAP3K15 search', 'MAP3K15,human', 'MAP3K15');
insert into pgpt_search (name, terms, gene) values('INHBE search', 'INHBE,human', 'INHBE');
insert into pgpt_search (name, terms, gene) values('GIGYF1 search', 'GIGYF1,human', 'GIGYF1');
insert into pgpt_search (name, terms, gene) values('GPR75 search', 'GPR75,human', 'GPR75');
  
insert into pgpt_search (name, terms, gene) values('SLC30A8 search', 'SLC30A8,human', 'SLC30A8');
insert into pgpt_keyword (keyword) values ('SLC30A8');
  
  
  
  
  

insert into pgpt_search_keyword (search_id, keyword_id) values(1, 1);
insert into pgpt_search_keyword (search_id, keyword_id) values(1, 2);



-- verify
select * from pgpt_paper_abstract pa, pgpt_paper_abstract pb 
where pa.id != pb.id and pa.pubmed_id = pb.pubmed_id;



-- count
select count(id) from pgpt_paper_abstract where pubmed_id is not null;

select count(sp.id), sp.search_id, se.terms
from pgpt_search_paper sp, pgpt_search se
where sp.search_id = se.id
group by search_id order by search_id;


select count(sp.id), sp.search_id, sp.document_level, se.terms
from pgpt_gpt_paper sp, pgpt_search se
where sp.search_id = se.id
group by search_id, sp.document_level 
order by search_id;




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


select count(distinct parent_id), document_level from pgpt_gpt_paper group by document_level order by document_level;

--- find papers for first level inference
select abst.id, abst.abstract 
from pgpt_paper_abstract abst, pgpt_search_paper seapaper 
where abst.document_level = 0 and seapaper.paper_id = abst.pubmed_id and seapaper.search_id = 1
and abst.id not in (select child_id from pgpt_gpt_paper where search_id = 1);

-- find gpt entries for following level inference
select abst.id, abst.abstract, abst.document_level
from pgpt_paper_abstract abst, pgpt_gpt_paper gpt
where abst.document_level = 1 and gpt.parent_id = abst.id and gpt.search_id = 4
and abst.id not in (select child_id from pgpt_gpt_paper where search_id = 4);


select * from pgpt_search_paper;
select abst.id, abst.abstract, abst.document_level
from {}.pgpt_paper_abstract abst, {}.pgpt_gpt_paper gpt
where abst.document_level = %s and gpt.parent_id = abst.id and gpt.search_id = %s
and abst.id not in (select child_id from {}.pgpt_gpt_paper where search_id = %s) limit %s

delete from pgpt_gpt_paper;
delete from pgpt_paper_abstract where pubmed_id is null and id not in (select distinct parent_id from pgpt_gpt_paper);

update pgpt_gpt_paper node
  join pgpt_paper_abstract paper on node.child_id = paper.pubmed_id
  set node.child_id = paper.id;


insert into pgpt_search_paper (search_id, paper_id) 
select distinct pubmed_id, 1 from pgpt_paper;

update pgpt_search_paper set search_id = paper_id, paper_id = search_id where search_id > 1000;


update pgpt_search set ready = 'N';
update pgpt_search set ready = 'Y' where id = 8;


-- data fixes
-- 20230721 - change search from 0 to 1 for PPARG for id 1 to 247
update pgpt_gpt_paper set search_id = 1;






select distinct abst.id, abst.document_level, abst.pubmed_id
from pgpt_paper_abstract abst, pgpt_gpt_paper gpt
where abst.document_level = 2 and gpt.parent_id = abst.id and gpt.search_id = 4
and abst.id not in (select child_id from pgpt_gpt_paper where search_id = 4);




