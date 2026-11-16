with bounds as (
    select
        min(data_referencia) as min_dt,
        max(data_referencia) as max_dt
    from {{ ref("gold_carga_energia_se") }}
),
calendar as (
    select cast(gs as date) as data_referencia
    from bounds,
    generate_series(min_dt, max_dt, interval 1 day) as t(gs)
)

select c.data_referencia
from calendar c
left join {{ ref("gold_carga_energia_se") }} g
    on c.data_referencia = g.data_referencia
where g.data_referencia is null
