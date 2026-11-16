select
    id_subsistema,
    data_referencia,
    carga_mwmed
from {{ ref("silver_carga_energia_diaria") }}
where carga_mwmed <= 0
