

create table gene_lookup as select * from richards_gene.gene_ncbi;

alter table gene_lookup add index glook_gene_idx (gene);

-- mysql> desc gene_lookup;
-- +---------+--------------+------+-----+---------+-------+
-- | Field   | Type         | Null | Key | Default | Extra |
-- +---------+--------------+------+-----+---------+-------+
-- | id      | int(11)      | NO   |     | 0       |       |
-- | gene    | varchar(100) | NO   |     | NULL    |       |
-- | ncbi_id | varchar(100) | YES  |     | NULL    |       |
-- +---------+--------------+------+-----+---------+-------+

