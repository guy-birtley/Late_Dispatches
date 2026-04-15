
# from sqlalchemy import text, create_engine
# import pandas as pd


# rufus_engine = create_engine(r"sqlite:///C:/Python Projects/local.db")

# #create phantom parts view
# with rufus_engine.connect() as conn:
#     print(pd.read_sql(f'''
#             SELECT stck.rufus_stkno_id AS raw_stkno_id, COALESCE(strc.rufus_component_id, stck.rufus_stkno_id) AS non_phantom_stkno_id
#             FROM stck
#             -- get phantom components (stock items)
#             LEFT JOIN strc ON strc.rufus_product_id = stck.rufus_stkno_id AND  stck.stck_user1 != ' ' AND stck.stck_prod_group != 10003
#             -- get stck info about components
#             LEFT JOIN stck phantom_comp ON phantom_comp.rufus_stkno_id = strc.rufus_component_id
#             WHERE (phantom_comp.stck_prod_group NOT LIKE '9%' -- filter phantom parts incorrectly set up against black products
#                 OR phantom_comp.stck_prod_group IS NULL)
#                 AND stck.rufus_stkno_id = 26003''', con = conn))




if (X.std(dim=-1) == 0).any():