

-- find all the successful gene summaries by run id
select search.id, search.gene, abs.gpt_run_id, abs.date_created
from pgpt_search search, pgpt_paper_abstract abs
where search.id = abs.search_top_level_of
and abs.gpt_run_id = 8
order by search.gene;

-- get searches that have not been done for a gpt run id
SELECT distinct search.id, search.gene, search.pubmed_count
FROM pgpt_search search
LEFT JOIN pgpt_paper_abstract abs
ON abs.search_top_level_of = search.id 
and abs.gpt_run_id = 8
WHERE abs.id IS NULL
order by search.id;

-- inverse query
SELECT distinct search.id, search.gene, search.pubmed_count
FROM pgpt_search search
LEFT JOIN pgpt_paper_abstract abs
ON abs.search_top_level_of = search.id 
and abs.gpt_run_id = 8
WHERE abs.id IS not NULL
order by search.id;

select count(distinct(search_top_level_of)) as number_done from pgpt_paper_abstract where gpt_run_id = 8;
