

-- count of number of genes with paper counts > 0
select count(id), avg(pubmed_count) from pgpt_search where pubmed_count > 0;

-- mysql> select count(id), avg(pubmed_count) from pgpt_search where pubmed_count > 0;
-- +-----------+-------------------+
-- | count(id) | avg(pubmed_count) |
-- +-----------+-------------------+
-- |     16404 |          665.9376 |
-- +-----------+-------------------+
-- 1 row in set (0.33 sec)


-- expect 
select count(id) from pgpt_search_paper;

