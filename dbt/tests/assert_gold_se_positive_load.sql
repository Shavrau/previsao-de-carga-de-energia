select
    data_referencia,
    carga_mwmed
from {{ ref("gold_carga_energia_se") }}
where carga_mwmed <= 0
