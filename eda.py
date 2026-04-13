from sqlalchemy import create_engine
import pandas as pd

count_monthly_dispatch_query = '''
WITH sub AS (
    SELECT acaud_sys_date, acaud_post_time / 10000
    FROM acaud
    JOIN stck ON acaud.rufus_stkno_id = stck.rufus_stkno_id AND stck.stck_prod_group IN (10001, 10005)
        AND acaud_qty != 0 AND acaud_option = 'PS2'
    GROUP BY acaud_sys_date, acaud_post_time / 10000, acaud.rufus_stkno_id
), Sub2 AS (
SELECT strftime('%Y', acaud_sys_date), strftime('%m', acaud_sys_date), count(acaud_sys_date) as trans
FROM sub
GROUP BY strftime('%Y', acaud_sys_date), strftime('%m', acaud_sys_date)
) SELECT avg(trans) FROM sub2
'''

count_monthly_orders_query = '''
WITH sub AS (
    SELECT sord_last_mod_date, sord_date_req
    FROM sord
    JOIN stck ON sord.rufus_stkno_id = stck.rufus_stkno_id AND stck.stck_prod_group IN (10001, 10005)
        AND sord_qty_req != 0
    GROUP BY sord_last_mod_date, sord_last_mod_time / 1000000, sord.rufus_stkno_id, sord_date_req
), Sub2 AS (
SELECT strftime('%Y', sord_last_mod_date), strftime('%m', sord_last_mod_date), count(sord_last_mod_date) as trans
FROM sub
GROUP BY strftime('%Y', sord_last_mod_date), strftime('%m', sord_last_mod_date)
) SELECT avg(trans) FROM sub2
'''

count_all_transactions_by_stkno = '''
WITH valid_ids AS (
    SELECT DISTINCT(acaud.rufus_stkno_id) AS rufus_stkno_id
    FROM acaud
    JOIN stck ON acaud.rufus_stkno_id = stck.rufus_stkno_id
    WHERE acaud_option = 'PS2' AND stck.stck_prod_group IN (10009) --, 10005)
        AND strftime('%Y', acaud_sys_date) = '2025'
),
sub AS (
    SELECT acaud_sys_date, acaud_post_time / 10000, acaud.rufus_stkno_id
    FROM acaud
    JOIN valid_ids ON valid_ids.rufus_stkno_id = acaud.rufus_stkno_id
    WHERE strftime('%Y', acaud_sys_date) IN ('2025', '2024')
    GROUP BY acaud_sys_date, acaud_post_time / 10000, acaud.rufus_stkno_id
), Sub2 AS (
SELECT rufus_stkno_id, count(rufus_stkno_id) as trans
FROM sub
GROUP BY rufus_stkno_id
HAVING count(rufus_stkno_id) > 2
), Quartiles AS (
SELECT
trans, rufus_stkno_id,
NTILE(4) OVER (ORDER BY trans) AS quartile_group
FROM sub2
)
SELECT
COUNT(trans) AS total_products, SUM(trans) AS total_transactions,
AVG(CASE WHEN trans > 256 THEN 1.0 ELSE 0.0 END) AS percentage_above_512,
MIN(trans) AS min,
MAX(CASE WHEN quartile_group = 1 THEN trans END) AS lower_quartile,
MAX(CASE WHEN quartile_group = 2 THEN trans END) AS mean,
MAX(CASE WHEN quartile_group = 3 THEN trans END) AS upper_quartile,
MAX(trans) AS max
FROM Quartiles;

'''

standard_prods_linked_to_non_standard_black_query = '''
SELECT strc.rufus_product_id, prod.stck_stkno AS prod_stkno, prod.stck_prod_group AS prod_group, strc.rufus_component_id, comp.stck_stkno AS comp_stkno, comp.stck_prod_group AS comp_group
FROM stck prod
JOIN strc ON prod.rufus_stkno_id = strc.rufus_product_id
JOIN stck comp ON comp.rufus_stkno_id = strc.rufus_component_id
WHERE prod.stck_prod_group = 10001 AND comp.stck_prod_group != 99
'''


query = count_all_transactions_by_stkno

rufus_engine = create_engine(r"sqlite:///C:/Python Projects/local.db")
with rufus_engine.connect() as conn:
    print(pd.read_sql(query, con=conn))