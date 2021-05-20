
drop table if exists agg_aggregator_phenotype;
create table agg_aggregator_phenotype (
  id                        int not null auto_increment primary key,
  phenotype_id              varchar(100) not null,
  phenotype_name            varchar(500),
  group_name                varchar(500)
);

alter table agg_aggregator_phenotype add unique index u_phenotype_id_idx (phenotype_id);


-- reports
select * from agg_aggregator_phenotype
where phenotype_id not in (
  select node_code from tran_test.comb_node_ontology where node_type_id in (1, 3)
)
order by phenotype_name;

