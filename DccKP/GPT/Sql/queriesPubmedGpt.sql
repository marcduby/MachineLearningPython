

-- find all the successful gene summaries by run id
select search.id, search.gene, abs.gpt_run_id, abs.date_created
from pgpt_search search, pgpt_paper_abstract abs
where search.id = abs.search_top_level_of
and abs.gpt_run_id = 8
order by search.gene;



