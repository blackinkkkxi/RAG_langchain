# RAG_langchain

检索增强生成（RAG）系统学习与实现项目，涵盖从基础到高级的 RAG 技术栈。

## 📋 目录

- [项目简介](#项目简介)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [核心模块](#核心模块)
- [学习路线](#学习路线)
- [进阶主题](#进阶主题)

## 项目简介

本项目系统地介绍了如何使用LangChain构建生产级 RAG 系统，包含：

- 📄 **文档处理**：多种格式文档加载与解析
- 🔪 **文本分割**：智能分块策略
- 🎯 **向量检索**：主流向量数据库对比与实现
- 🤖 **LLM 集成**：OpenAI、DeepSeek 等模型支持
- 🔁 **Reranker**：检索结果重排序优化
- 📊 **评估体系**：RAG 系统效果评估
- 🧠 **Agent 系统**：多 Agent 协作框架

## 快速开始

### 基础 RAG 系统

运行 [rag_chat.ipynb](rag_chat.ipynb) 快速搭建一个基于 OpenAI 的 RAG 系统：

```python
# 加载文档 -> 分割 -> 嵌入 -> 检索 -> 生成
```

### DeepSeek RAG

参考 [rag_deepseekR1.ipynb](rag_deepseekR1.ipynb) 了解如何使用 DeepSeek 模型构建 RAG。

## 项目结构

```
RAG_langchain/
├── agent/                 # Agent 框架实现
│   ├── multi_agent/      # 多 Agent 协作系统
│   └── tool_call/        # 工具调用实现
├── vectorDB/             # 向量数据库相关
│   ├── metadatafilter.ipynb
│   └── flat_search.ipynb
├── learn/                # 核心学习模块
│   ├── doc_loader/       # 文档加载器
│   ├── text_splitter/    # 文本分割策略
│   ├── embedding_model/  # 嵌入模型
│   ├── reranker/         # 重排序模型
│   ├── evaluation/       # 评估方法
│   ├── advanced_method/  # 高级检索方法
│   └── Long-Context-RAG/ # 长上下文 RAG
├── chunsize/             # 分块大小优化
├── graph_rag/            # Graph RAG 实现
├── RLM/                  # RLM 相关
├── data/                 # 示例数据
└── *.ipynb               # 主要实现 notebook
```

## 核心模块

### 📄 文档加载 (Document Loader)

- **基础加载**：[learn/doc_loader/PDF_loader.ipynb](learn/doc_loader/PDF_loader.ipynb)
  - PyPDFLoader、UnstructuredPDFLoader 等基础加载器
- **进阶解析**：[learn/doc_loader/pdf_parse.ipynb](learn/doc_loader/pdf_parse.ipynb)
  - 处理复杂 PDF 格式（表格、多列布局等）

### 🔪 文本分割 (Text Splitter)

- **基础分割**：[learn/text_splitter/textspliter_ex.ipynb](learn/text_splitter/textspliter_ex.ipynb)
  - CharacterTextSplitter、RecursiveCharacterTextSplitter
- **可视化分割**：理解分割效果

### 🎯 嵌入模型 (Embedding)

- **模型选择**：[learn/embedding_model/](learn/embedding_model/)
  - 如何选择合适的 Embedding 模型？
  - OpenAI 可变维度 Embedding
  - **多模态检索**：[learn/embedding_model/Gemini_Embedding_2_Multimodal_Retrieval.ipynb](learn/embedding_model/Gemini_Embedding_2_Multimodal_Retrieval.ipynb)
  - 使用 Gemini Embedding 2 实现 Multimodal Retrieval
- **测试对比**：[embedding_test/](embedding_test/)
- **测试对比**：[embedding_test/](embedding_test/)

### 🗄️ 向量数据库 (Vector Database)

- **基础检索**：[vectorDB/](vectorDB/)
- **元数据过滤**：[metadatafilter.ipynb](vectorDB/metadatafilter.ipynb)
- **向量库对比**：Chroma vs FAISS vs Milvus

### 🔁 检索优化 (Reranker)

- **Reranker 模型**：[learn/reranker/](learn/reranker/)
  - 什么是 Reranker？
  - 如何在 RAG 中使用 Reranker？

### 📊 系统评估 (Evaluation)

- **评估指标**：[learn/evaluation/](learn/evaluation/)
  - 准确率、召回率、响应质量等指标
  - RAGAS等评估框架

## 学习路线

### 🌟 初学者路线

1. **快速体验**：运行 [rag_chat.ipynb](rag_chat.ipynb) 了解 RAG 基本流程
2. **文档处理**：学习 [doc_loader](learn/doc_loader/) 掌握文档加载
3. **文本分割**：学习 [text_splitter](learn/text_splitter/) 理解分块策略
4. **嵌入模型**：学习 [embedding_model](learn/embedding_model/) 选择合适的模型

### 🚀 进阶路线

5. **向量检索**：深入研究 [vectorDB/](vectorDB/) 不同向量库的特点
6. **检索优化**：学习 [reranker](learn/reranker/) 提升检索精度
7. **系统评估**：学习 [evaluation](learn/evaluation/) 建立评估体系
8. **高级方法**：探索 [advanced_method](learn/advanced_method/)
9. **RLM**：[RLM/](RLM/) 递归LLM


## 进阶主题

### 🤖 Agent 系统

- **工具调用**：[agent/tool_call/](agent/tool_call/)
  - 基于 Qwen 的 Function Calling
- **多 Agent 协作**：[agent/multi_agent/](agent/multi_agent/)
  - AutoGen、LangGraph 等框架

### 📈 优化策略

- **分块优化**：[chunsize/](chunsize/) 最佳分块大小探索
- **元数据过滤**：[vectorDB/metadatafilter.ipynb](vectorDB/metadatafilter.ipynb)
- **混合检索**：稠密检索 + 稀疏检索


# 从零构建一个Mini Claude Code

- **初级完整流程**： [mini_claude_code/](mini_claude_code/mini_claude_code.py)
- **TODO**



## 🌟 Star History

如果你觉得这个项目对你有帮助，请给个 Star 支持一下！

[![Star History Chart](https://api.star-history.com/svg?repos=blackinkkkxi/RAG_langchain&type=Date)](https://star-history.com/#blackinkkkxi/RAG_langchain&Date)