

-- count of number of genes with paper counts > 0
select count(id), avg(pubmed_count) from pgpt_search where pubmed_count > 0;

-- mysql> select count(id), avg(pubmed_count) from pgpt_search where pubmed_count > 0;
-- +-----------+-------------------+
-- | count(id) | avg(pubmed_count) |
-- +-----------+-------------------+
-- |     16404 |          665.9376 |
-- +-----------+-------------------+
-- 1 row in set (0.33 sec)


-- expect 11^e6
select count(id) from pgpt_search_paper;

-- mysql> select count(id) from pgpt_search_paper;
-- +-----------+
-- | count(id) |
-- +-----------+
-- |   8696654 |
-- +-----------+
-- 1 row in set (0.38 sec)


-- create a table to compare the counts
create database verify_gpt;

drop table verify_gpt.search_paper_link_count_20230923;
create table verify_gpt.search_paper_link_count_20230923 as
select count(ref.id) calc_pub_count, search.pubmed_count, ref.search_id, search.gene 
from pgpt_search_paper ref, pgpt_search search
where ref.search_id = search.id 
group by ref.search_id, search.pubmed_count, search.gene
order by search.gene;

select * from verify_gpt.search_paper_link_count_20230923 where calc_pub_count != pubmed_count order by gene;

