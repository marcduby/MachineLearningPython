

create table gene_lookup as select * from richards_gene.gene_ncbi;

alter table gene_lookup add index glook_gene_idx (gene);
alter table gene_lookup add index glook_ncbi_idx (ncbi_id);

-- mysql> desc gene_lookup;
-- +---------+--------------+------+-----+---------+-------+
-- | Field   | Type         | Null | Key | Default | Extra |
-- +---------+--------------+------+-----+---------+-------+
-- | id      | int(11)      | NO   |     | 0       |       |
-- | gene    | varchar(100) | NO   |     | NULL    |       |
-- | ncbi_id | varchar(100) | YES  |     | NULL    |       |
-- +---------+--------------+------+-----+---------+-------+



create table gene_lookup (
    id int(11),
    gene varchar(100),
    ncbi_id varchar(200)
)