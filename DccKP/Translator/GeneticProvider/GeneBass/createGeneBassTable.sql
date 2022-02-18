
drop table if exists data_genebass_gene_phenotype;
create table data_genebass_gene_phenotype (
  id                        int not null auto_increment primary key,
  gene                      varchar(100) not null,                          -- gene_symbol
  gene_ncbi_id              varchar(100),                             -- FROM JOINED LOOKUP TABLE
  phenotype_genebass        varchar(1000),                                  -- disease_title
  phenotype_ontology_id     varchar(100),                                   -- disease_curie
  phenotype_genepro_name    varchar(1000),                            -- FROM JOINED LOOKUP TABLE OR NODE NORMALIZER
  gene_genepro_id           int(9),
  phenotype_genepro_id      int(9),
  pheno_num_genebass        int(9),
  pheno_coding_genebass     int(9),
  pvalue                    double,                                    -- 
  standard_error            double,                                    -- 
  beta                      double,                                    -- 
  abf                       double,                                    -- 
  probability               double,                                    -- 
  score_genepro             double                                    -- classification calculated
);



