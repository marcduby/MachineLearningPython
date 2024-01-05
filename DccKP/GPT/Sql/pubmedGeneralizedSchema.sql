

-- pubmed abstract table
drop table if exists pubm_paper_abstract;
create table pubm_paper_abstract (
  pubmed_id                 int(9) not null primary key,
  title                     varchar(1000) null,
  abstract                  varchar(4000) null,
  journal_name              varchar(2000) null,
  in_pubmed_file            varchar(50) null,
  paper_year                int(9) null,
  count_reference           int(9) default 0 not null,
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
);
alter table pubm_paper_abstract add index pubm_pap_abs_pub (pubmed_id);

-- files processed table
drop table if exists pubm_file_processed;
create table pubm_file_processed (
  id                        int not null auto_increment primary key,
  file_name                 varchar(100) not null,
  process_name              varchar(100) not null,
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
);
alter table pubm_file_processed add index pubm_fil_pro_fil (file_name);
alter table pubm_file_processed add index pubm_fil_pro_pro (process_name);




-- keywords tables
drop table if exists pgpt_keyword;
create table pgpt_keyword (
  id                        int not null auto_increment primary key,
  keyword                   varchar(5000) not null,
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
);


-- -- paper table
-- drop table if exists pgpt_paper;
-- create table pgpt_paper (
--   pubmed_id                 int not null primary key,
--   to_download               enum('Y', 'N') default 'Y' not null,
--   download_success          enum('Y', 'N') default 'N' not null,
--   count_reference           int(9) default 0 not null,
--   date_created              datetime DEFAULT CURRENT_TIMESTAMP
-- );

-- alter table pgpt_paper add column to_download enum('Y', 'N') default 'Y' not null;
-- alter table pgpt_paper add column download_success enum('Y', 'N') default 'N' not null;
-- alter table pgpt_paper add column count_reference int(9) default 0 not null;


-- paper/search table
-- paper_id is pubmed_id
drop table if exists pgpt_search_paper;
create table pgpt_search_paper (
  id                        int not null auto_increment primary key,
  search_id                 int(9) not null,
  pubmed_id                 int(9) not null,
  date_created              datetime DEFAULT CURRENT_TIMESTAMP
);
alter table pgpt_search_paper add index pgpt_ser_pap_ser (search_id);
alter table pgpt_search_paper add index pgpt_ser_pap_pap (paper_id);



