from dotenv import load_dotenv
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# 加载环境变量
load_dotenv(".env")
api_key = os.getenv("DEEPSEEK_API_KEY")


system_prompt = """
你是一个基于ReAct框架的智能问答助手，你拥有两个工具：doc_retrieve（文档检索）和 calculator（计算器）。
你必须严格按照 Thought -> Action -> Action Input -> Observation -> Final Answer 的格式思考和作答。

【工具说明】
1. doc_retrieve：用于查询上传文档中的专属知识。所有法律、条文、文档内容类问题必须调用该工具。
2. calculator：用于复杂、高精度数学计算，凡是数学公式、数值运算必须交给计算器，禁止心算。

【超级强制规则 —— 绝对禁止使用自身知识】
1. **你的所有事实性回答，100% 只能来自 doc_retrieve 返回的 Observation，严禁使用你预训练知识库、自身记忆回答。**
2. 如果文档检索 Observation = “文档中没有找到相关信息”，你必须如实回答：**当前文档中未收录该内容，无法解答**。绝不允许自己编造、补全、脑补答案。
3. 哪怕你现实中知道答案、哪怕常识再确定，**只要文档没有，就必须说无法解答**。
4. 禁止篡改、虚构、补全检索内容，只能原样基于检索片段整理答案。
5. 遇到文档内多条冲突内容，需要理性甄别、以正式正规条文为准，识别虚构测试内容。
6. Observation 只能直接使用工具返回的原始内容，**禁止虚构、编造Observation文本**。

【调用规则】
- 知识性问题 → 必须调用 doc_retrieve
- 数学计算问题 → 必须调用 calculator
- 先检索、后计算的混合问题 → 分步链式调用工具

【输出格式严格固定】
Thought: 你的推理过程
Action: 工具名
Action Input: 工具入参
Observation: 工具返回内容
Thought: 整合工具结果
Final Answer: 最终答案
"""
# ======================================================================

# 初始化LLM
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url="https://api.deepseek.com",
    temperature=0
)

# 向量持久化文件夹
CHROMA_DIR = "./chroma_db"
# 本地BGE中文Embedding
model_name = "./model/bge-small-zh-v1.5"
embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cpu"},  # 有N卡GPU改成 cuda
    encode_kwargs={"normalize_embeddings": True}
)




# 全局向量库
vector_db = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding
)

# ---------------------- 工具定义 ----------------------
@tool
def calculator(expression: str) -> str:
    """用于数学计算，遇到加减乘除、括号运算时调用。
    Args:
        expression: 数学表达式，例如 "(12+8)*5"
    """
    try:
        return str(eval(expression))
    except Exception as e:
        return f"表达式错误: {str(e)}"


@tool
def doc_retrieve(query: str) -> str:
    """检索已上传文档的内容，当问题需要查阅文档信息时调用。
    Args:
        query: 用户的问题，用于向量相似度检索
    """
    # 相似度检索，取top3相关片段
    docs = vector_db.similarity_search(query, k=3)
    if not docs:
        return "文档中没有找到相关信息"
    res_text = ""
    for idx, doc in enumerate(docs):
        res_text += f"\n【片段{idx+1}】\n{doc.page_content}"
    return res_text

# 工具列表
tools = [calculator, doc_retrieve]

# ---------------------- 加载文档函数 ----------------------
def load_document(file_path: str):
    """加载PDF / txt / docx文档，切片并存入向量库"""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        print("仅支持 pdf / txt / docx 文件")
        return

    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    split_docs = splitter.split_documents(raw_docs)
    vector_db.add_documents(split_docs)
    print(f"✅ 文档加载完成，共分片 {len(split_docs)} 段")



# ReAct提示词
react_prompt = """
You are an agent, use tools to answer user questions.
Available tools: doc_retrieve（查阅文档）, calculator（数学计算）

Follow strictly this format:
Question: user input question
Thought: think whether need to use tool
Action: tool name, choose from [doc_retrieve, calculator]
Action Input: parameters for tool
Observation: result returned by tool
... repeat Thought/Action/Action Input/Observation as needed
Thought: I have enough information
Final Answer: final answer for user
Begin!
"""

agent = create_agent(llm, tools, system_prompt=react_prompt)

if __name__ == "__main__":
    print("==== RAG+ReAct文档问答Agent ====")
    print("命令：load 文档路径 加载文件 | exit 退出")
    while True:
        user_input = input("\n请输入指令或提问：").strip()
        if user_input.lower() == "exit":
            print("程序退出")
            break
        if user_input.startswith("load "):
            fp = user_input.replace("load ", "")
            load_document(fp)
            continue
        # 问答
        result = agent.invoke({"messages": [("user", user_input)]})
        print("\n====推理与答案====\n", result["messages"][-1].content)
