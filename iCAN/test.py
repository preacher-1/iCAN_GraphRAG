import os
from dotenv import load_dotenv
import asyncio

import pandas as pd
import tiktoken

from graphrag.query.indexer_adapters import read_indexer_entities, read_indexer_reports
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.structured_search.global_search.search import GlobalSearch


load_dotenv()
api_key = os.environ.get("GRAPHRAG_API_KEY")
llm_model = os.environ.get("GRAPHRAG_LLM_MODEL")

llm = ChatOpenAI(
    api_key=api_key,
    model=llm_model,
    # api_type=OpenaiApiType.OpenAI,  # OpenaiApiType.OpenAI or OpenaiApiType.AzureOpenAI
    api_base="https://api.deepseek.com/v1/",
    max_retries=20,
)

token_encoder = tiktoken.get_encoding("cl100k_base")


# 从索引管道生成的 Parquet 文件
INPUT_DIR = "output/20240802-173526/artifacts"
COMMUNITY_REPORT_TABLE = "create_final_community_reports"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"

# 在Leiden社区层次结构中，我们将从中加载社区报告的社区级别，更高的值意味着我们使用来自更细粒度社区的报告（代价是更高的计算成本）
COMMUNITY_LEVEL = 2

entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")

reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
# print(f"Report records: {len(report_df)}")
# print(report_df.head())

# 根据社区报告构建全局上下文
context_builder = GlobalCommunityContext(
    community_reports=reports,
    entities=entities,  # 如果您不想使用社区权重进行排名，则默认为 None
    token_encoder=token_encoder,
)

# 执行全局搜索
context_builder_params = {
    "use_community_summary": False,  # False means using full community reports. True means using community short summaries.
    "shuffle_data": True,
    "include_community_rank": True,
    "min_community_rank": 0,
    "community_rank_name": "rank",
    "include_community_weight": True,
    "community_weight_name": "occurrence weight",
    "normalize_community_weight": True,
    "max_tokens": 12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
    "context_name": "Reports",
}

map_llm_params = {
    "max_tokens": 1000,
    "temperature": 0.0,
    "response_format": {"type": "json_object"},
}

reduce_llm_params = {
    "max_tokens": 2000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000-1500)
    "temperature": 0.0,
}

search_engine = GlobalSearch(
    llm=llm,
    context_builder=context_builder,
    token_encoder=token_encoder,
    max_data_tokens=12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
    map_llm_params=map_llm_params,
    reduce_llm_params=reduce_llm_params,
    allow_general_knowledge=False,  # 将此设置为 True 将添加指令，以鼓励 LLM 在响应中融入一般知识，这可能会增加幻觉，但在某些用例中可能有用。
    json_mode=True,  # set this to False if your LLM model does not support JSON mode.
    context_builder_params=context_builder_params,
    concurrent_coroutines=32,
    response_type="multiple paragraphs",  # 描述响应类型和格式的自由格式文本，可以是任何内容，例如优先级列表、单个或多个段落、多页报告
)

result = search_engine.search(
    "What are the most common topics or subjects in all these articles? Only return topics and descriptions in json format."
)
print(result.response)

# 检查用于构建 LLM 响应上下文的数据
# print(result.context_data["reports"])

# 检查 LLM 调用和令牌的数量
# print(f"LLM calls: {result.llm_calls}. LLM tokens: {result.prompt_tokens}")
