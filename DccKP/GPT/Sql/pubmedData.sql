


-- add run data
-- insert into pgpt_gpt_run (name, gpt_engine_id, prompt, to_process)
-- values('20230821 ChatGPT genetics', 2, 'as a genetics researcher, summarize the genetics of gene {} from the following text: \n{}', 'Y');

insert into pgpt_gpt_run (name, gpt_engine_id, prompt, to_process)
values('20230821 Paid ChatGPT genetics', 2, 
  'Below are the abstracts from different research papers on gene {}. Please read through the abstracts and write a 200 word summary that synthesizes the key findings of the papers on the genetics of gene {}\n{}', 
  'Y');

-- 7
select search.gene, abstract.search_top_level_of, abstract.gpt_run_id, abstract.abstract 
from pgpt_paper_abstract abstract, pgpt_search search 
where abstract.gpt_run_id = 7 and abstract.search_top_level_of is not null and abstract.search_top_level_of = search.id;

-- 8
insert into pgpt_gpt_run (name, gpt_engine_id, prompt, to_process)
values('20230821 Paid ChatGPT biology', 2, 
  'Below are the abstracts from different research papers on gene {}. Please read through the abstracts and as a genetics researcher write a 100 word summary that synthesizes the key findings of the papers on the biology of gene {}\n{}', 
  'Y');
update pgpt_gpt_run set prompt = 
  'Below are the abstracts from different research papers on gene {}. Please read through the abstracts and as a genetics researcher write a 200 word summary that synthesizes the key findings of the papers on the biology of gene {}\n{}', 
where id = 8;

-- 9
insert into pgpt_gpt_run (name, gpt_engine_id, prompt, to_process, max_docs_per_level)
values('20230826 Paid ChatGPT genetics - no abstracts', 2, 
  'Write a 200 word summary that synthesizes the key findings of the papers on the genetics of gene {}', 
  'Y', 0);

-- 10, 12
insert into pgpt_gpt_run (name, gpt_engine_id, prompt, to_process, max_docs_per_level)
values('20230826 Paid ChatGPT genetics - no abstracts', 2, 
  'Write a 200 word summary that synthesizes the key findings on the genetics of gene {}', 
  'Y', 0);

-- 11, 13, 14
insert into pgpt_gpt_run (name, gpt_engine_id, prompt, to_process, max_docs_per_level)
values('20230826 Paid ChatGPT biology - no abstracts', 2, 
  'As a genetics researcher write a 100 word summary that synthesizes the key findings on the biology of gene {}', 
  'Y', 0);

-- 16
insert into pgpt_gpt_run (name, gpt_engine_id, prompt, to_process, max_docs_per_level)
values('20230829 Paid ChatGPT generic', 2, 
  'Below are the abstracts from different research papers on gene {}. Please read through the abstracts and write a 200 word summary that synthesizes the key findings of the papers on gene {}\n{}', 
  'Y', 50);

-- 17
insert into pgpt_gpt_run (name, gpt_engine_id, prompt, to_process, max_docs_per_level)
values('20230829 Paid ChatGPT generic - no abstracts', 2, 
  'Write a 200 word summary that synthesizes the key findings on gene {}', 
  'Y', 0);


