\# DocumentQA：基于ReAct框架的文档问答智能体

本项目实现ReAct智能体 + BGE本地向量RAG，支持文档检索、计算器双工具调用，可对上传文档进行问答。



\## 功能

\- ReAct推理链路 Thought -> Action -> Observation -> Final Answer

\- BGE嵌入 + Chroma向量数据库本地文档向量化

\- 双工具：文档检索、数学计算，支持链式调用

\- Prompt约束，限制模型仅使用文档内知识，缓解大模型幻觉与知识逃逸



\## 环境

Python >=3.10

```bash

pip install -r requirements.txt



