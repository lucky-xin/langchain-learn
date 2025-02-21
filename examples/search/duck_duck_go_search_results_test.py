from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.wikidata.tool import WikidataQueryRun
from langchain_community.utilities.wikidata import WikidataAPIWrapper

# 使用默认设置创建DuckDuckGo搜索工具
search_duck_duck_go = DuckDuckGoSearchResults(num_results=5)
search_wikidata = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())

query = "2024年奥运会100米自由泳决赛冠军是谁？他的家乡是哪里？请用中文答"

results = search_duck_duck_go.run(query)
print(results)
print("===========================================================================================================")
results = search_wikidata.run(query)
print(results)
