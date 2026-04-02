# 🧧 AI 成语接龙系统（RAG 智能版）

一个基于 **LangChain + FAISS + Ollama(Qwen) + Tkinter GUI** 的智能成语接龙小游戏项目。

本项目结合 **RAG（检索增强生成）技术** 和 **本地大语言模型**，实现了一个可以与用户实时进行成语接龙互动的桌面应用。

---

## 📌 项目简介

该项目使用本地成语词库构建向量数据库，通过语义检索匹配相关成语，再结合大语言模型进行智能生成，实现高质量成语接龙。

相比传统规则匹配方式，本项目引入 **RAG 检索增强生成架构**，使 AI 接龙更加智能、准确、可扩展。

---

## 🚀 项目功能

### 🎯 核心功能
- 用户输入四字成语
- 自动校验成语合法性
- 基于上一成语最后一个字智能接龙
- 本地向量检索匹配候选成语
- 大模型生成最优结果
- GUI 实时对话交互

---

## 🧠 技术架构

```text
用户输入
   ↓
Tkinter 图形界面
   ↓
成语合法性校验
   ↓
FAISS 向量检索
   ↓
LangChain RAG Chain
   ↓
Qwen 本地大模型生成
   ↓
返回 AI 接龙结果
```

---

## 🛠 技术栈

### 后端 / AI
- Python
- LangChain
- FAISS
- HuggingFace Embeddings
- Ollama
- Qwen 1.8B

### GUI
- Tkinter
- 多线程 Threading

### 向量模型
- BGE Large Chinese Embedding
- `bge-large-zh-v1.5`

---

## 📂 项目结构

```text
idiom-rag-game/
│
├── main.py                # 主程序
├── idiom.txt              # 成语词库
├── README.md
└── models/
    └── bge-large-zh-v1.5
```

---

## ⚙️ 核心实现逻辑

---

### 1️⃣ 成语库加载

从本地 txt 文件读取所有四字成语：

```python
def load_all_idioms(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    return [line.strip() for line in lines if len(line.strip()) == 4]
```

---

### 2️⃣ 向量数据库构建

使用 `FAISS` 构建本地向量检索库：

```python
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_PATH)
vs = FAISS.from_documents(chunks, embedding)
retriever = vs.as_retriever(search_kwargs={"k": 15})
```

---

### 3️⃣ RAG 检索增强生成

结合检索结果和用户输入构建 Prompt：

```python
prompt = ChatPromptTemplate.from_template("""
只许接四字成语。
用上一成语最后一字开头。
只能用下面的成语：
{context}

上一个：{question}
你接：
""")
```

---

### 4️⃣ 本地大模型调用

通过 Ollama 调用本地 Qwen 模型：

```python
chat_model = ChatOpenAI(
    openai_api_key="ollama",
    base_url="http://localhost:11434/v1",
    model="qwen:1.8b"
)
```

---

### 5️⃣ GUI 交互界面

使用 Tkinter 实现桌面聊天式界面：

- 输入框
- 对话窗口
- 发送按钮
- 多线程防卡顿

---

## 🎨 项目亮点

---

### ⭐ RAG 技术落地
将 **检索增强生成（RAG）** 应用于小游戏场景，提升回答准确率。

---

### ⭐ 本地部署 LLM
无需调用云端 API，完全本地运行：

- 数据安全
- 响应速度快
- 零额外调用成本

---

### ⭐ 向量检索优化
使用 FAISS 提高成语召回效率，避免大模型胡乱生成。

---

### ⭐ 工程化 GUI
具有完整桌面交互界面，适合作为完整项目展示。

---

## 💡 项目运行

---

### 安装依赖

```bash
pip install langchain
pip install langchain-community
pip install langchain-openai
pip install langchain-huggingface
pip install faiss-cpu
pip install tkinter
```

---

### 启动 Ollama

```bash
ollama run qwen:1.8b
```

---

### 运行项目

```bash
python main.py
```

---

## 📈 项目优化方向

未来可继续优化：

- 成语释义展示
- 拼音匹配
- 难度模式
- AI 记忆历史对局
- Web 版本（Gradio / Streamlit）
- FastAPI 接口部署
- 多模型切换

---

## 📝 简历项目描述（可直接使用）

**基于 LangChain + FAISS + Ollama 的 RAG 智能成语接龙系统**

- 基于 RAG 架构实现智能成语接龙桌面应用
- 使用 FAISS 构建本地向量数据库，实现高效语义检索
- 基于 Ollama 本地部署 Qwen 大模型进行生成
- 使用 Tkinter 实现完整桌面 GUI 交互界面
- 引入多线程机制优化用户交互体验
- 提升 AI 回答准确率与响应速度

---

## 👨‍💻 作者

Shuo Z
