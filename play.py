

from sqlalchemy import text, create_engine
import pandas as pd
import numpy as np


engine = create_engine(r"sqlite:///C:/Python Projects/local.db")

query = '''
WITH sub AS (SELECT s.rufus_stkno_id, sord_stkno, sord_order, max(sode_desp_date) AS sode_desp_date, max(post_datetime) AS acaud_desp_date
            FROM sord s
            LEFT JOIN acaud a ON s.rufus_stkno_id = a.rufus_stkno_id AND a.acaud_ref1 = s.sord_order
            LEFT JOIN sode ON s.rufus_stkno_id = sode.rufus_stkno_id AND s.sord_order = sode.sode_order
            WHERE sord_date_req BETWEEN '2024-01-01' AND '2025-12-31'
                AND (acaud_sys_date IS NULL) -- AND sode_desp_date IS NOT NULL
            GROUP BY s.rufus_stkno_id, sord_stkno, sord_order)
            -- HAVING max(acaud_sys_date) IS NULL) -- max(sode_desp_date) != max(acaud_sys_date) OR
SELECT stck_prod_group, stck_user1, sub.*
FROM sub
JOIN stck ON stck.rufus_stkno_id = sub.rufus_stkno_id
WHERE stck_prod_group = 10001 -- AND stck_user1 = ' '
        '''

query = '''
SELECT ps.non_phantom_stkno_id AS stkno_id,
        sord_date_req AS req_date,
        sord_order_date AS order_date,
        sord_qty_req AS qty,
        sord_order AS order_num,
        MAX(acaud_sys_date) AS desp_date,
        MAX(acaud_sys_date) > sord_date_req AS late
FROM sord s
JOIN phantom_stknos ps ON ps.raw_stkno_id = s.rufus_stkno_id
JOIN acaud a ON a.acaud_ref1 = s.sord_order
JOIN phantom_stknos pa ON pa.raw_stkno_id = a.rufus_stkno_id AND pa.non_phantom_stkno_id = ps.non_phantom_stkno_id
GROUP BY ps.non_phantom_stkno_id, sord_date_req, sord_order_date, sord_qty_req, sord_order
'''
# query = '''
# WITH sord_desp_date AS (
#             SELECT s.rufus_stkno_id, s.sord_date_req, s.sord_order_date, s.sord_qty_req,
#                     max(acaud_sys_date) AS desp_date -- get max despatch date for each order
#             FROM sord s
#             JOIN acaud a ON s.rufus_stkno_id = a.rufus_stkno_id AND a.acaud_ref1 = s.sord_order
#             GROUP BY s.rufus_stkno_id, s.sord_date_req, s.sord_order_date, s.sord_qty_req
#         )
#         SELECT non_phantom_stkno_id AS stkno_id,
#                 sord_date_req AS req_date,
#                 sord_order_date AS order_date,
#                 MAX(desp_date) AS desp_date,
#                 SUM(sord_qty_req) AS qty,
#                 MAX(desp_date) > sord_date_req AS late
#         FROM sord_desp_date s
#         JOIN phantom_stknos p ON p.raw_stkno_id = s.rufus_stkno_id
#         WHERE sord_qty_req > 0
#         GROUP BY stkno_id, sord_date_req, sord_order_date
# '''

query = 'select * from sord limit 0'
with engine.connect() as conn:
    print(pd.read_sql(query, con = conn).columns)
