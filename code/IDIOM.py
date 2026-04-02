from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. 先把所有成语读到列表里（用于校验）
def load_all_idioms(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    return [line.strip() for line in lines if len(line.strip()) == 4]

idiom_list = load_all_idioms(r"E:\big model use\idiom\code\idiom.txt")

# 2. 模型配置（低温、短输出）
chat_model = ChatOpenAI(
    openai_api_key="ollama",
    base_url="http://localhost:11434/v1",
    model="gemma3:1b",
    temperature=0.01,
    max_tokens=8
)

# 3. RAG 向量库
loader = TextLoader(r"E:\big model use\idiom\code\idiom.txt", encoding='utf-8')
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
chunks = text_splitter.split_documents(docs)

embedding = HuggingFaceEmbeddings(
    model_name=r'E:\moodel\ma\idiom\txt\idiom.txt'
)
vs = FAISS.from_documents(chunks, embedding)
retriever = vs.as_retriever(search_kwargs={"k": 15})

# 4. 极简到变态的提示词
prompt = ChatPromptTemplate.from_template("""
只许接四字成语。
用上一成语最后一字开头。
只能用下面的成语：
{context}

上一个：{question}
你接：
""")

def format_docs(docs):
    return "\n".join([d.page_content.strip() for d in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | chat_model
    | StrOutputParser()
)

# ------------------- 游戏主逻辑 -------------------
print("=== RAG + gemma3:1b 成语接龙===")

while True:
    user_idiom = input("你出：").strip()
    if len(user_idiom) != 4 and user_idiom not in idiom_list:
        print("请输入四字成语！")
        continue

    last_char = user_idiom[-1]

    # 调用 RAG + 模型
    raw_ans = rag_chain.invoke(user_idiom).strip()

    # 暴力清洗：只留4个汉字
    ai_idiom = ''.join([c for c in raw_ans if '\u4e00' <= c <= '\u9fff'])[:4]

    # 强制校验：必须以 last_char 开头 + 必须在成语表里
    if len(ai_idiom) == 4 and ai_idiom.startswith(last_char) and ai_idiom in idiom_list:
        print("AI：", ai_idiom)
    else:
        # 从本地列表找一个合规的（兜底，防止模型抽风）
        for idiom in idiom_list:
            if idiom.startswith(last_char):
                print("AI：", idiom)
                break
        else:
            print("AI：我接不上了，你赢了！")
            break