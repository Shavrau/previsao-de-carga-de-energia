select
    id_subsistema,
    data_referencia,
    count(*) as total_registros
from {{ ref("silver_carga_energia_diaria") }}
group by 1, 2
having count(*) > 1
