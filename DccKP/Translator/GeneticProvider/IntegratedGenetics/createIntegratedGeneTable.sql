

-- mysql> desc MAGMA_GENES;
-- +----------+--------------+------+-----+---------+-------+
-- | Field    | Type         | Null | Key | Default | Extra |
-- +----------+--------------+------+-----+---------+-------+
-- | ID       | varchar(255) | YES  | MUL | NULL    |       |
-- | GENE     | varchar(255) | YES  | MUL | NULL    |       |
-- | DISEASE  | varchar(255) | YES  | MUL | NULL    |       |
-- | CATEGORY | varchar(255) | YES  |     | NULL    |       |
-- | PVALUE   | double       | YES  | MUL | NULL    |       |
-- +----------+--------------+------+-----+---------+-------+
-- 5 rows in set (0.00 sec)

-- mysql> desc SCORE_GENES;
-- +----------+--------------+------+-----+---------+-------+
-- | Field    | Type         | Null | Key | Default | Extra |
-- +----------+--------------+------+-----+---------+-------+
-- | ID       | varchar(255) | YES  | MUL | NULL    |       |
-- | GENE     | varchar(255) | YES  |     | NULL    |       |
-- | DISEASE  | varchar(255) | YES  | MUL | NULL    |       |
-- | CATEGORY | varchar(255) | YES  |     | NULL    |       |
-- | SCORE    | double       | YES  | MUL | NULL    |       |
-- +----------+--------------+------+-----+---------+-------+
-- 5 rows in set (0.00 sec)

-- mysql> 
-- mysql> 


alter table SCORE_GENES add index scoregene_gene_idx (GENE);
