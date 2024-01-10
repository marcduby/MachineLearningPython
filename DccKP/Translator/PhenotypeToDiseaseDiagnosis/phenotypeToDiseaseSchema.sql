

-- phenotype to disease link
drop table if exists phen_phenotype_disease_link;
create table phen_phenotype_disease_link (
  id                        int not null auto_increment primary key,
  phenotype_curie           varchar(100) not null,
  disease_curie             varchar(100) not null,
  phenotype_name            varchar(100) null,
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
);
alter table phen_phenotype_disease_link add index phen_phe_dis_dis (disease_curie);
alter table phen_phenotype_disease_link add index phen_phe_dis_phe (phenotype_curie);


-- disease table
drop table if exists phen_disease;
create table phen_disease (
  id                        int not null auto_increment primary key,
  curie                     varchar(100) not null,
  biolink_curie             varchar(100) null,
  name                      varchar(500) null,
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
);
alter table phen_disease add index phen_dis_cur (curie);




