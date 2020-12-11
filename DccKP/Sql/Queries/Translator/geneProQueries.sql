

-- get the ordered phenotypes in DB
-- select pheno.phenotype_code, pheno.phenotype, pheno.tran_efo_id as efo_id 

select pheno.phenotype, pheno.tran_efo_id as efo_id
from MAGMA_GENES mag, phenotype_lookup pheno 
where mag.disease = pheno.tran_efo_id
group by pheno.phenotype;

select pheno.phenotype, pheno.tran_efo_id as efo_id
from MAGMA_PATHWAYS mag, phenotype_lookup pheno 
where mag.disease = pheno.tran_efo_id
group by pheno.phenotype;

select pheno.phenotype, pheno.tran_efo_id as efo_id
from richards_gene mag, phenotype_lookup pheno 
where mag.phenotype = pheno.tran_efo_id
group by pheno.phenotype;

select pheno.phenotype, pheno.tran_efo_id as efo_id
from SCORE_GENES mag, phenotype_lookup pheno 
where mag.disease = pheno.tran_efo_id
group by pheno.phenotype;

select pheno.phenotype, pheno.tran_efo_id as efo_id
from abc_gene_phenotype mag, phenotype_lookup pheno 
where mag.phenotype_efo_id = pheno.tran_efo_id
group by pheno.phenotype;


-- scratch
drop table if exists phenotype_lookup;
create table phenotype_lookup as select * from genetics_lookup.phenotype_lookup;


