

-- fix the pgpt_search.pubmed_count inflated by pubmed api returning all papers for join of one term with no results
select * from pgpt_search where pubmed_count > 20000000;

update pgpt_search set pubmed_count = 0 where pubmed_count > 20000000;

-- verify searches with no result counts
select count(id) from pgpt_search where pubmed_count < 1;

-- delete from search/paper link  table any search with no counts
delete ref from pgpt_search_paper ref
inner join pgpt_search search on search.id = ref.search_id
where search.pubmed_count < 1;

-- pre count
-- mysql> select count(id) from pgpt_search_paper;
-- +-----------+
-- | count(id) |
-- +-----------+
-- |   8696654 |
-- +-----------+
-- 1 row in set (4.98 sec)

-- post count
-- mysql> select count(id) from pgpt_search_paper;
-- +-----------+
-- | count(id) |
-- +-----------+
-- |   4817042 |
-- +-----------+
-- 1 row in set (43.76 sec)



