{{
    config(
        materialized="table",
        post_hook="COPY (SELECT * FROM {{ this }} ORDER BY id_subsistema, data_referencia) TO 'data/gold/carga_energia_subsistemas.parquet' (FORMAT PARQUET)"
    )
}}

select
    data_referencia,
    id_subsistema,
    nom_subsistema,
    carga_mwmed,
    case
        when data_referencia < date '2021-03-01' then 'pre_2021_03'
        when data_referencia < date '2023-04-29' then 'nao_despachadas_2021_2023'
        else 'com_mmgd_pos_2023_04'
    end as regime_metodologico
from {{ ref("silver_carga_energia_diaria") }}
order by id_subsistema, data_referencia
