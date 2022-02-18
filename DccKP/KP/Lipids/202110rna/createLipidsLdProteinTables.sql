

drop table if exists lipids_ld_protein_u20s;
create table lipids_ld_protein_u20s (
  id                        int not null auto_increment primary key,
  gene                      varchar(100) not null,
  csn_value                 float
);



drop table if exists lipids_ld_protein_huh7;
create table lipids_ld_protein_huh7 (
  id                        int not null auto_increment primary key,
  gene                      varchar(100) not null,
  csn_value                 float
);


drop table if exists lipids_ld_protein_rnai;
create table lipids_ld_protein_rnai (
  id                        int not null auto_increment primary key,
  gene_source               varchar(100) not null,
  gene_target               varchar(100) not null,
  rho_value                 float
);
create index lps_rnai_src on lipids_ld_protein_rnai(gene_source);
create index lps_rnai_tgt on lipids_ld_protein_rnai(gene_target);
create index lps_rnai_val on lipids_ld_protein_rnai(rho_value);


drop table if exists lipids_ld_protein_csn;
create table lipids_ld_protein_csn (
  id                        int not null auto_increment primary key,
  gene                      varchar(100) not null,
  csn_u20s_value            float,
  csn_huh7_value            float
);
create index lps_csn_gen on lipids_ld_protein_csn(gene);



-- populate the csn table
insert into lipids_ld_protein_csn (gene) select gene from lipids_ld_protein_u20s where gene not in (select gene from lipids_ld_protein_csn);
insert into lipids_ld_protein_csn (gene) select gene from lipids_ld_protein_huh7 where gene not in (select gene from lipids_ld_protein_csn);

update lipids_ld_protein_csn csn inner join lipids_ld_protein_u20s u20s
on csn.gene = u20s.gene
set csn.csn_u20s_value = u20s.csn_value;

update lipids_ld_protein_csn csn inner join lipids_ld_protein_huh7 huh7
on csn.gene = huh7.gene
set csn.csn_huh7_value = huh7.csn_value;





-- scratch
select * from lipids_ld_protein_huh7 csn1, lipids_ld_protein_huh7 csn2 
where csn1.gene = csn2.gene and csn1.id != csn2.id;

select * from lipids_ld_protein_u20s csn1, lipids_ld_protein_u20s csn2 
where csn1.gene = csn2.gene and csn1.id != csn2.id;


-- mysqldump -u root -p lipids_load lipids_ld_protein_csn lipids_ld_protein_rnai > 20211012lipidsLdProteinDataDump.sql  




