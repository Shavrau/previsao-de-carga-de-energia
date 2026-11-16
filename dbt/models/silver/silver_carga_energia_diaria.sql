{{
    config(
        materialized="table"
    )
}}

with bronze as (
    select
        trim(cast(id_subsistema as varchar)) as id_subsistema,
        trim(cast(nom_subsistema as varchar)) as nom_subsistema,
        cast(din_instante as date) as data_referencia,
        cast(val_cargaenergiamwmed as double) as carga_mwmed
    from {{ source("bronze", "carga_energia_diaria") }}
)

select
    id_subsistema,
    nom_subsistema,
    data_referencia,
    carga_mwmed
from bronze
where data_referencia is not null
  and carga_mwmed is not null
qualify row_number() over (
    partition by id_subsistema, data_referencia
    order by data_referencia
) = 1
