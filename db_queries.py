import pandas as pd
from sqlalchemy import text, create_engine
import pickle
from helper import prod_groups, tprint

rufus_engine = create_engine(r"sqlite:///C:/Python Projects/local.db")

#create phantom parts view
with rufus_engine.connect() as conn:
    conn.execute(text('DROP VIEW IF EXISTS phantom_stknos'))
    conn.execute(text(f'''
        CREATE VIEW phantom_stknos AS
            SELECT stck.rufus_stkno_id AS raw_stkno_id, MIN(COALESCE(strc.rufus_component_id, stck.rufus_stkno_id)) AS non_phantom_stkno_id
            FROM stck
            -- get phantom components (stock items)
            LEFT JOIN strc ON strc.rufus_product_id = stck.rufus_stkno_id AND  stck.stck_user1 != ' ' AND stck.stck_prod_group != 10003
            -- get stck info about components
            LEFT JOIN stck phantom_comp ON phantom_comp.rufus_stkno_id = strc.rufus_component_id
            WHERE (phantom_comp.stck_prod_group IN {prod_groups} -- filter phantom parts incorrectly set up against black products
                OR phantom_comp.stck_prod_group IS NULL)
                AND stck.stck_prod_group IN {prod_groups} -- all relevant product groups
            GROUP BY stck.rufus_stkno_id
        '''))
    tprint('reading order data')
    orders = pd.read_sql(
# f'''
# WITH sord_desp_date AS (
#     SELECT s.rufus_stkno_id, s.sord_date_req, s.sord_order_date, s.sord_qty_req,
#             max(acaud_sys_date) AS desp_date -- get max despatch date for each order
#     FROM sord s
#     JOIN acaud a ON s.rufus_stkno_id = a.rufus_stkno_id AND a.acaud_ref1 = s.sord_order
#     GROUP BY s.rufus_stkno_id, s.sord_date_req, s.sord_order_date, s.sord_qty_req
# )
# SELECT non_phantom_stkno_id AS stkno_id,
#         sord_date_req AS req_date,
#         sord_order_date AS order_date,
#         MAX(desp_date) AS desp_date,
#         SUM(sord_qty_req) AS qty,
#         MAX(desp_date) > sord_date_req AS late
# FROM sord_desp_date s
# JOIN phantom_stknos p ON p.raw_stkno_id = s.rufus_stkno_id
# WHERE sord_qty_req > 0
# GROUP BY stkno_id, sord_date_req, sord_order_date
# ''', 
    '''
SELECT ps.non_phantom_stkno_id AS stkno_id,
        sord_date_req AS req_date,
        sord_order_date AS order_date, -- or last mod date?
        sord_last_mod_date AS last_mod_date,
        sord_datetime AS last_mod_datetime,
        sord_qty_req AS qty,
        sord_order AS order_num,
        MAX(acaud_sys_date) AS desp_date,
        MAX(post_datetime) AS desp_datetime,
        MAX(acaud_sys_date) > sord_date_req AS late
FROM sord s
JOIN phantom_stknos ps ON ps.raw_stkno_id = s.rufus_stkno_id
JOIN phantom_stknos pa ON pa.non_phantom_stkno_id = ps.non_phantom_stkno_id
JOIN acaud a ON a.rufus_stkno_id = pa.raw_stkno_id AND a.acaud_ref1 = s.sord_order
WHERE sord_qty_req > 0
    AND sord_last_mod_date >= sord_order_date
    AND sord_date_req >= sord_order_date
GROUP BY ps.non_phantom_stkno_id, sord_date_req, sord_order_date, sord_qty_req, sord_order
''', con = conn, index_col = 'stkno_id')
    #slight descrepency here as partially fulfilled orders will still show full allocation until despatched but fuck it
    tprint('reading trans data')
    trans = pd.read_sql(f'''
        WITH ranked AS (
            SELECT
                CASE WHEN (acaud_job LIKE '%STCK%' OR acaud_job LIKE '%STOCK%') AND acaud_job NOT LIKE '%TAKE%' THEN 1 ELSE 0 END AS correction,
                acaud_sys_date,
                acaud.rufus_stkno_id,
                stck_prod_group,
                acaud_qty,
                acaud_open_balance,
                ROW_NUMBER() OVER (
                    PARTITION BY acaud_sys_date, acaud.rufus_stkno_id, acaud_qty>0
                    ORDER BY acaud_post_time DESC
                ) AS row_num
            FROM acaud
            JOIN stck ON stck.rufus_stkno_id = acaud.rufus_stkno_id AND stck_prod_group IN {prod_groups + (99, 99999)} -- need 99999 because of eda standard_prods_linked_to_non_standard_black_query
        ),
        grouped AS (
            SELECT
                correction,
                acaud_sys_date AS trans_date,
                rufus_stkno_id,
                stck_prod_group,
                SUM(acaud_qty) AS qty,
                MAX(CASE WHEN row_num = 1 THEN acaud_open_balance + acaud_qty END) AS on_hand -- closing balance of the day
            FROM ranked
            GROUP BY acaud_sys_date, rufus_stkno_id, acaud_qty > 0, stck_prod_group, correction
        ),
        wip_and_finished AS (
        -- finished transations
        SELECT
            correction,
            trans_date,
            rufus_stkno_id AS stkno_id,
            qty,
            on_hand,
            NULL AS wip_on_hand,
            0 AS wip
        FROM grouped
        WHERE stck_prod_group IN {prod_groups}
                        
        UNION ALL
        
        -- wip transactions
        SELECT
            0 AS correction,
            trans_date,
            MIN(strc.rufus_product_id) AS stkno_id,
            qty,
            NULL AS on_hand,
            on_hand AS wip_on_hand,
            1 AS wip
        FROM grouped
        JOIN strc ON strc.rufus_component_id = grouped.rufus_stkno_id
        JOIN stck fg ON strc.rufus_product_id = fg.rufus_stkno_id AND fg.stck_prod_group IN {prod_groups}  -- see note above (could be just = 99 if aligned properly)
        WHERE grouped.stck_prod_group IN (99, 99999)
        GROUP BY trans_date, qty, on_hand, grouped.rufus_stkno_id -- in case black part linked to more than 1 white part
        )
                        
        SELECT *
        FROM wip_and_finished
        ORDER BY stkno_id, trans_date

    ''', con = conn, index_col='stkno_id')
    tprint('reading stck data')
    stck = pd.read_sql(f'''
        SELECT rufus_stkno_id,
            stck_dimension_length,
            stck_size,
            stck_user_check01,stck_user_check02, stck_user_check03,
            stck_prod_group
        FROM stck
        WHERE stck_prod_group IN {prod_groups}
        GROUP BY rufus_stkno_id -- incase double entries of same product
    ''', index_col= 'rufus_stkno_id', con = conn)

tprint('saving')
stck.to_pickle(r'cache\stck_df.pkl')
orders.to_pickle(r'cache\orders_df.pkl')
trans.to_pickle(r'cache\trans_df.pkl')