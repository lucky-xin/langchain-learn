from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, URL, util

url = URL(

)
engine = create_engine(
    url=url,
    pool_recycle=3600,
    echo=True
)

db = SQLDatabase(
    engine=engine,
    include_tables=["ref_old_brand", "ref_old_model", "ref_old_basictrim", "ref_old_city"],  # 白名单过滤
    sample_rows_in_table_info=2  # 在提示词中展示的示例数据行数
)