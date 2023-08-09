
-- search table
drop table if exists pgpt_search;
create table pgpt_search (
  id                        int not null auto_increment primary key,
  name                      varchar(500) not null,
  terms                     varchar(5000) null,
  gene                      varchar(100) null,
  pubmed_count              int(7) default -1 not null,
  ready                     enum('Y', 'N') default 'N' not null,
  to_download               enum('Y', 'N') default 'N' not null,
  to_download_ids           enum('Y', 'N') default 'N' not null,
  date_last_download        datetime null,
  date_last_summary         datetime null,
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
);

-- alter table pgpt_search add column ready enum('Y', 'N') default 'N' not null;
-- alter table pgpt_search add column to_download enum('Y', 'N') default 'N' not null;
-- alter table pgpt_search add column date_last_download datetime null;
-- alter table pgpt_search add column pubmed_count int(7) default -1 not null;
-- alter table pgpt_search add column date_last_summary datetime null;
-- alter table pgpt_search add column to_download_ids enum('Y', 'N') default 'N' not null;


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
-- paper_id is pubmed_id
drop table if exists pgpt_search_paper;
create table pgpt_search_paper (
  id                        int not null auto_increment primary key,
  search_id                 int(9) not null,
  paper_id                  int(9) not null,
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
);
alter table pgpt_search_paper add index pgpt_ser_pap_ser (search_id);
alter table pgpt_search_paper add index pgpt_ser_pap_pap (paper_id);

ALTER TABLE pgpt_search_paper CHANGE paper_id pubmed_id int(9) not null;

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
  gpt_run_id                int(9) null,
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
);
alter table pgpt_paper_abstract add index pgpt_pap_abs_pub (pubmed_id);

alter table pgpt_paper_abstract add column search_top_level_of int(9) null;
alter table pgpt_paper_abstract add column gpt_run_id int(9) null;

-- add first run data
update pgpt_paper_abstract set gpt_run_id = 1 where pubmed_id is null;

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


-- paper reference table
drop table if exists pgpt_paper_reference;
create table pgpt_paper_reference (
  id                        int not null auto_increment primary key,
  pubmed_id                 int(9) not null,
  ref_pubmed_id             int(9) not null,
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
);
alter table pgpt_paper_reference add index pgpt_pap_ref_pib (pubmed_id);
alter table pgpt_paper_reference add index pgpt_pap_ref_ref (ref_pubmed_id);


-- gpt engine table
drop table if exists pgpt_gpt_engine;
create table pgpt_gpt_engine (
  id                        int not null primary key,
  gpt_name                  varchar(200) not null,
  gpt_description           varchar(2000) null,
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
);

insert into pgpt_gpt_engine (id, gpt_name) values(1, 'ChatGPT 3.5 free service');
insert into pgpt_gpt_engine (id, gpt_name) values(2, 'ChatGPT 3.5 paid service');
insert into pgpt_gpt_engine (id, gpt_name) values(3, 'Claude 2 free service');
insert into pgpt_gpt_engine (id, gpt_name) values(4, 'Llama2 7B chat HF local');


-- gpt run tableupdate pgpt_paper_abstract set gpt_run_id = 1 where pubmed_id is null;

drop table if exists pgpt_gpt_run;
create table pgpt_gpt_run (
  id                        int not null auto_increment primary key,
  name                      varchar(200) not null,
  gpt_engine_id             int(9) not null,
  prompt                    varchar(4000) null,
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
);

insert into pgpt_gpt_run (name, gpt_engine_id) values('ChatGPT 3.5 free service', 1);
insert into pgpt_gpt_run (name, gpt_engine_id) values('ChatGPT 3.5 paid service', 2);
insert into pgpt_gpt_run (name, gpt_engine_id) values('Llama2 on g5xl AWS', 3);




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
  
insert into pgpt_search (name, terms, gene) values('GCK search', 'GCK,human', 'GCK');
insert into pgpt_keyword (keyword) values ('GCK');
  
  
insert into pgpt_search_keyword (search_id, keyword_id) values(1, 1);
insert into pgpt_search_keyword (search_id, keyword_id) values(1, 2);

-- lipodystrophy genes
insert into pgpt_search (name, terms, gene) values('LMNA search', 'LMNA,human', 'LMNA');
insert into pgpt_keyword (keyword) values ('LMNA');

insert into pgpt_search (name, terms, gene) values('PLIN1 search', 'PLIN1,human', 'PLIN1');
insert into pgpt_keyword (keyword) values ('PLIN1');

insert into pgpt_search (name, terms, gene) values('AGPAT2 search', 'AGPAT2,human', 'AGPAT2');
insert into pgpt_keyword (keyword) values ('AGPAT2');

insert into pgpt_search (name, terms, gene) values('BSCL2 search', 'BSCL2,human', 'BSCL2');
insert into pgpt_keyword (keyword) values ('BSCL2');

insert into pgpt_search (name, terms, gene) values('CAV1 search', 'CAV1,human', 'CAV1');
insert into pgpt_keyword (keyword) values ('CAV1');

insert into pgpt_search (name, terms, gene) values('PTRF search', 'PTRF,human', 'PTRF');
insert into pgpt_keyword (keyword) values ('PTRF');

-- mody genes
insert into pgpt_search (name, terms, gene) values('HNF1B search', 'HNF1B,human', 'HNF1B');
insert into pgpt_keyword (keyword) values ('HNF1B');
  
insert into pgpt_search (name, terms, gene) values('CEL search', 'CEL,human', 'CEL');
insert into pgpt_keyword (keyword) values ('CEL');
  
insert into pgpt_search (name, terms, gene) values('PDX1 search', 'PDX1,human', 'PDX1');
insert into pgpt_keyword (keyword) values ('PDX1');
  
insert into pgpt_search (name, terms, gene) values('INS search', 'INS,human', 'INS');
insert into pgpt_keyword (keyword) values ('INS');
  
insert into pgpt_search (name, terms, gene) values('NEUROD1 search', 'NEUROD1,human', 'NEUROD1');
insert into pgpt_keyword (keyword) values ('NEUROD1');
  
insert into pgpt_search (name, terms, gene) values('KLF11 search', 'KLF11,human', 'KLF11');
insert into pgpt_keyword (keyword) values ('KLF11');
  


-- verify
select * from pgpt_paper_abstract pa, pgpt_paper_abstract pb 
where pa.id != pb.id and pa.pubmed_id = pb.pubmed_id;









-- count
select count(id) from pgpt_paper_abstract where pubmed_id is not null;

select count(sp.id), se.* from pgpt_search_paper sp, pgpt_search se where sp.search_id = se.id group by search_id order by search_id;


select count(sp.id), sp.search_id, sp.document_level, se.terms
from pgpt_gpt_paper sp, pgpt_search se
where sp.search_id = se.id
group by search_id, sp.document_level 
order by search_id;

select search_top_level_of from pgpt_paper_abstract where search_top_level_of is not null order by search_top_level_of;


-- query
-- get the search results
select se.gene, abst.abstract
from pgpt_paper_abstract abst, pgpt_search se 
where abst.search_top_level_of = se.id
order by se.gene;


select * from pgpt_paper_abstract where search_top_level_of = 1;

select se.id, se.gene, abs.abstract
from pgpt_paper_abstract abs, pgpt_search se 
where se.id = abs.search_top_level_of
order by se.id;


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

select avg(wordcount(abstract)) from pgpt_paper_abstract where id < 100;
SELECT
    ROUND (   
        (
            CHAR_LENGTH(abstract) - CHAR_LENGTH(REPLACE (abstract, " ", "")) 
        ) 
        / CHAR_LENGTH(" ")        
    ) AS count    
FROM from pgpt_paper_abstract where id < 10;


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

search_top_level_of



-- update searches
update pgpt_search set ready = 'N';
-- update pgpt_search set ready = 'Y' where id = 1;
update pgpt_search set ready = 'Y' where id not in (1, 2, 4, 8);

-- updates
update pgpt_search set to_download = 'Y' where id > 20 and id < 30;

update pgpt_search set date_last_summary = sysdate() -1 where id in (select search_top_level_of from pgpt_paper_abstract where search_top_level_of is not null);

update pgpt_search set ready = 'N';
update pgpt_search set ready = 'Y' where pubmed_count < 900 and date_last_summary is null and date_last_download is not null;

select * from pgpt_search where pubmed_count < 900 and date_last_summary is null and date_last_download is not null;
select * from pgpt_search where date_last_download is not null;

select * from pgpt_search where ready = 'Y';

-- data fixes
-- 20230721 - change search from 0 to 1 for PPARG for id 1 to 247
update pgpt_gpt_paper set search_id = 1;






select distinct abst.id, abst.document_level, abst.pubmed_id
from pgpt_paper_abstract abst, pgpt_gpt_paper gpt
where abst.document_level = 2 and gpt.parent_id = abst.id and gpt.search_id = 4
and abst.id not in (select child_id from pgpt_gpt_paper where search_id = 4);


-- load genes from t2d huge scores of translator
-- insert genes
insert into pgpt_keyword (keyword)
select phe.gene_code
from tran_upkeep.agg_gene_phenotype phe
where phe.phenotype_code = 'T2D'
and phe.gene_code not in (select gene from pgpt_search) 
order by phe.abf_probability_combined desc limit 1000;

-- insert searches
insert into pgpt_search (name, terms, gene) 
select concat(phe.gene_code, ' search'), concat(phe.gene_code, ',human'), phe.gene_code
from tran_upkeep.agg_gene_phenotype phe
where phe.phenotype_code = 'T2D'
and phe.gene_code not in (select gene from pgpt_search) 
order by phe.abf_probability_combined desc limit 1000;

insert into pgpt_search (name, terms, gene) 
select concat(phe.gene_code, ' search'), concat(phe.gene_code, ',human'), phe.gene_code
from tran_upkeep.agg_gene_phenotype phe
where phe.phenotype_code = 'AF'
and phe.gene_code not in (select gene from pgpt_search) 
order by phe.abf_probability_combined desc limit 500;


select phe.gene_code
from tran_upkeep.agg_gene_phenotype phe
where phe.phenotype_code = 'T2D'
and phe.gene_code not in (select gene from pgpt_search) 
order by phe.abf_probability_combined desc limit 100;



insert into pgpt_search (name, terms, gene) values('GCK search', 'GCK,human', 'GCK');


insert into pgpt_keyword (keyword) values ('GCK');


-- gene sets
-- mody genes
select * from pgpt_search where gene in ('GCK', 'HNF1A', 'HNF1B', 'CEL', 'PDX1', 'HNF4A', 'INS', 'NEUROD1', 'KLF11') order by gene;


-- lipodystrophy genes
select * from pgpt_search where gene in ('LMNA', 'PPARG', 'PLIN1', 'AGPAT2', 'BSCL2', 'CAV1', 'PTRF') order by id;

select count(sp.id), sp.search_id, sp.document_level, se.gene, se.pubmed_count
from pgpt_gpt_paper sp, pgpt_search se
where sp.search_id = se.id and ((se.id = 1) or (se.id >= 137 and se.id <=142))
group by search_id, sp.document_level 
order by search_id;


select count(sp.id), sp.search_id, sp.document_level
from pgpt_gpt_paper sp, pgpt_search se
where sp.search_id = se.id
group by search_id, sp.document_level;


select count(sp.id)
from pgpt_gpt_paper sp, pgpt_search se
where sp.search_id = se.id and se.id = 137 and sp.document_level = 1;


select sp.abstract, se.gene 
from pgpt_paper_abstract sp, pgpt_search se
where sp.search_top_level_of = se.id and ((se.id = 1) or (se.id >= 137 and se.id <= 142));


