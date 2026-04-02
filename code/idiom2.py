import tkinter as tk
from tkinter import ttk, scrolledtext, END
import threading
from tkinter import font

# ------------------- RAG 核心依赖 -------------------
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ------------------- 全局配置 -------------------
IDIOM_PATH = r"E:\moodel\ma\idiom\txt\idiom.txt"
EMBEDDING_PATH = r"E:\big model use\code\learn\models\bge-large-zh-v1.5"

# ------------------- 1. 加载成语库 -------------------
def load_all_idioms(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
        return [line.strip() for line in lines if len(line.strip()) == 4]
    except:
        return []

idiom_list = load_all_idioms(IDIOM_PATH)

# ------------------- 2. RAG 向量库初始化 -------------------
loader = TextLoader(IDIOM_PATH, encoding='utf-8')
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
chunks = text_splitter.split_documents(docs)

embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_PATH)
vs = FAISS.from_documents(chunks, embedding)
retriever = vs.as_retriever(search_kwargs={"k": 15})

# ------------------- 3. AI 模型 -------------------
chat_model = ChatOpenAI(
    openai_api_key="ollama",
    base_url="http://localhost:11434/v1",
    model="qwen:1.8b",
    temperature=0.01,
    max_tokens=8
)

# ------------------- 4. 提示词 & 链 -------------------
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

# ------------------- 5. 游戏逻辑 -------------------
def generate_ai_idiom(user_idiom):
    last_char = user_idiom[-1]
    try:
        raw_ans = rag_chain.invoke(user_idiom).strip()
        ai_idiom = ''.join([c for c in raw_ans if '\u4e00' <= c <= '\u9fff'])[:4]
        if len(ai_idiom) == 4 and ai_idiom.startswith(last_char) and ai_idiom in idiom_list:
            return ai_idiom
    except:
        pass

    for idiom in idiom_list:
        if idiom.startswith(last_char):
            return idiom
    return None

# ------------------- 🌟 美化版界面 -------------------
class IdiomGameGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🧧 AI 成语接龙 - RAG 智能版")
        self.root.geometry("650x550")
        self.root.resizable(False, False)
        self.root.configure(bg="#F8F9FA")

        # 字体
        self.title_font = font.Font(family="微软雅黑", size=18, weight="bold")
        self.text_font = font.Font(family="微软雅黑", size=12)
        self.input_font = font.Font(family="微软雅黑", size=13)

        # 标题栏
        title_frame = tk.Frame(root, bg="#FFFFFF", padx=20, pady=12)
        title_frame.pack(fill="x", padx=20, pady=10)
        title_label = tk.Label(
            title_frame, text="🧧 智能成语接龙",
            font=self.title_font, bg="#FFFFFF", fg="#2D3748"
        )
        title_label.pack()

        # 对话区域
        chat_frame = tk.Frame(root, bg="#FFFFFF", padx=10, pady=10)
        chat_frame.pack(fill="both", expand=True, padx=20, pady=5)
        self.chat_box = scrolledtext.ScrolledText(
            chat_frame, width=70, height=18,
            font=self.text_font,
            bg="#F9FAFB", fg="#2D3748",
            relief="flat",
            wrap=tk.WORD
        )
        self.chat_box.pack(fill="both", expand=True)
        self.chat_box.insert(END, "🎯 请输入四字成语开始游戏……\n\n")
        self.chat_box.config(state=tk.DISABLED)

        # 输入栏
        input_frame = tk.Frame(root, bg="#F8F9FA", padx=20, pady=10)
        input_frame.pack(fill="x", padx=20, pady=5)
        self.input_entry = tk.Entry(
            input_frame, font=self.input_font,
            relief="flat", bg="#FFFFFF", fg="#2D3748",
            highlightthickness=1, highlightcolor="#D1D5DB"
        )
        self.input_entry.pack(side="left", fill="x", expand=True, ipady=6)
        self.input_entry.bind("<Return>", self.start_game)

        # 发送按钮
        self.send_btn = tk.Button(
            input_frame, text="发送", command=self.start_game,
            font=self.input_font,
            bg="#4F46E5", fg="white",
            relief="flat", activebackground="#4338CA",
            padx=16, pady=4, cursor="hand2"
        )
        self.send_btn.pack(side="right", padx=(8, 0))

    def append_msg(self, msg):
        self.chat_box.config(state=tk.NORMAL)
        self.chat_box.insert(END, msg + "\n")
        self.chat_box.see(END)
        self.chat_box.config(state=tk.DISABLED)

    def start_game(self, event=None):
        user_text = self.input_entry.get().strip()
        self.input_entry.delete(0, END)

        if len(user_text) != 4:
            self.append_msg("❌ 请输入标准四字成语！")
            return
        if user_text not in idiom_list:
            self.append_msg("❌ 该成语不在词库中，请换一个～")
            return

        self.append_msg(f"🧑 你：{user_text}")

        def task():
            ai_idiom = generate_ai_idiom(user_text)
            if ai_idiom:
                self.append_msg(f"🤖 AI：{ai_idiom}")
            else:
                self.append_msg("🤖 AI：我接不上啦，你赢了！🎉")
            self.append_msg("—" * 40)

        threading.Thread(target=task, daemon=True).start()

# ------------------- 启动 -------------------
if __name__ == "__main__":
    if not idiom_list:
        print("❌ 成语文件加载失败！")
    else:
        print(f"✅ 成语库加载完成：{len(idiom_list)} 个")
        root = tk.Tk()
        app = IdiomGameGUI(root)
        root.mainloop()