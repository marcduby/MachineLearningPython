
drop table if exists agg_aggregator_phenotype;
create table agg_aggregator_phenotype (
  id                        int not null auto_increment primary key,
  phenotype_id              varchar(100) not null,
  phenotype_name            varchar(500),
  group_name                varchar(500),
  ontology_id               varchar(100),
  in_translator             enum('true', 'false') default 'false',
  just_added_in             enum('true', 'false') default 'false',
  last_updated              timestamp default CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

alter table agg_aggregator_phenotype add unique index u_phenotype_id_idx (phenotype_id);



-- workflow
update tran_upkeep.agg_aggregator_phenotype set in_translator = 'true'
where phenotype_id COLLATE utf8mb4_general_ci in (
  select node_code from tran_test_202208.comb_node_ontology where node_type_id in (1, 3)
);


-- reports
select * from agg_aggregator_phenotype
where phenotype_id COLLATE utf8mb4_general_ci not in (
  select node_code from tran_test_202208.comb_node_ontology where node_type_id in (1, 3)
)
order by phenotype_name;



select * from agg_aggregator_phenotype
where phenotype_id COLLATE utf8mb4_general_ci in (
  select node_code from tran_test_202208.comb_node_ontology where node_type_id in (1, 3)
)
order by phenotype_name;


select count(id), in_translator from tran_upkeep.agg_aggregator_phenotype group by in_translator;

select * from tran_upkeep.agg_aggregator_phenotype where ontology_id is not null order by phenotype_name;

select id, phenotype_id, phenotype_name from tran_upkeep.agg_aggregator_phenotype 
where ontology_id is null and in_translator = 'false'
order by phenotype_name;
