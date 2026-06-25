from dotenv import load_dotenv
import os
import json
import math
import re
import time
import atexit
from openai import OpenAI
from pathlib import Path
from code_agent_tools import CodeTools, TOOL_MAP

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
model = os.getenv("LLM_MODEL")
root = os.getenv("PROJECT_ROOT")


# -------------------- 本地 Embedding 引擎（bge-large-zh-v1.5）--------------------
_embed_model = None  # 懒加载单例


def _get_embed_model():
    """懒加载 BGE 模型，只加载一次"""
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    try:
        from sentence_transformers import SentenceTransformer
        print("[Embed] 正在加载本地模型 BAAI/bge-large-zh-v1.5 ...")
        _embed_model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
        print("[Embed] 模型加载完成")
        return _embed_model
    except ImportError:
        print("[Embed] 未安装 sentence-transformers，回退关键词匹配")
        print("[Embed] 安装命令: pip install sentence-transformers")
        return None
    except Exception as e:
        print(f"[Embed] 模型加载失败: {e}，回退关键词匹配")
        return None


def _embed_text(text: str) -> list[float] | None:
    """用本地 BGE 模型将文本转为向量"""
    m = _get_embed_model()
    if m is None:
        return None
    try:
        # BGE 模型建议对 query 加前缀以提升效果
        return m.encode(text, normalize_embeddings=True).tolist()
    except Exception as e:
        print(f"[Embed] 编码失败: {e}")
        return None


# -------------------- RAG 记忆系统 --------------------
class RAGMemory:
    """基于 bge-large-zh-v1.5 本地 embedding 的长期记忆，带写入去重 + 懒保存"""

    STOP_WORDS = set("的了吗呢吧啊是和在也都很要去说会对为及与或但而".replace(" ", ""))
    DEDUP_THRESHOLD = 0.88     # 相似度超过此值视为重复
    SAVE_BATCH = 5             # 每 N 次 add 写一次盘
    SAVE_INTERVAL = 30         # 或每 30 秒写一次盘

    def __init__(self, storage_path: str = ".agent_memory.json"):
        self.storage_path = Path(storage_path)
        self.items: list[dict] = []
        self._id_counter = 0
        self._has_embedder = False
        self._dirty = False              # 是否有未持久化的变更
        self._add_since_save = 0         # 上次保存后新增条数
        self._last_save_time = time.time()
        self._dedup_skipped = 0          # 统计跳过的重复条数
        self.load()
        atexit.register(self._on_exit)   # 退出时兜底保存

    # ---------- 关键词分词（回退用）----------
    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        tokens = re.findall(r"[一-鿿]+|[a-zA-Z]+|\d+", text.lower())
        return [t for t in tokens if t not in cls.STOP_WORDS and len(t) > 1]

    @classmethod
    def _keyword_score(cls, query: str, text: str) -> float:
        q_tokens = cls._tokenize(query)
        t_tokens = cls._tokenize(text)
        if not q_tokens:
            return 0.0
        q_set = set(q_tokens)
        t_set = set(t_tokens)
        intersection = q_set & t_set
        union = q_set | t_set
        jaccard = len(intersection) / len(union) if union else 0.0
        q_bigrams = set(zip(q_tokens, q_tokens[1:])) if len(q_tokens) >= 2 else set()
        t_bigrams = set(zip(t_tokens, t_tokens[1:])) if len(t_tokens) >= 2 else set()
        bigram_score = len(q_bigrams & t_bigrams) / len(q_bigrams) if q_bigrams else 0.0
        coverage = len(intersection) / len(q_set)
        return 0.35 * jaccard + 0.25 * bigram_score + 0.40 * coverage

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    # ---------- 去重判断 ----------
    def _find_duplicate(self, query: str, answer: str, emb: list[float] | None) -> bool:
        """检查是否与已有记忆高度重复，只看最近的 10 条"""
        text = query + " " + answer
        candidates = self.items[-10:]  # 只看最近 10 条，O(10) 足够
        for item in candidates:
            if emb is not None and item.get("embedding"):
                score = self._cosine(emb, item["embedding"])
            else:
                score = self._keyword_score(text, item["query"] + " " + item["answer"])
            if score >= self.DEDUP_THRESHOLD:
                return True
        return False

    # ---------- 存取接口 ----------
    def add(self, query: str, answer: str, force: bool = False):
        """写入一条记忆，自动去重 + 懒保存。force=True 跳过去重"""
        text = f"问: {query}\n答: {answer}"
        emb = _embed_text(text) if self._has_embedder else None

        if not force and self._find_duplicate(query, answer, emb):
            self._dedup_skipped += 1
            return  # 重复，跳过

        item = {
            "id": self._id_counter,
            "query": query,
            "answer": answer,
            "embedding": emb,
            "timestamp": time.time()
        }
        self._id_counter += 1
        self.items.append(item)
        self._dirty = True
        self._add_since_save += 1

        if self._add_since_save >= self.SAVE_BATCH or \
           (time.time() - self._last_save_time) >= self.SAVE_INTERVAL:
            self.save()

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        if not self.items:
            return []

        if self._has_embedder:
            q_emb = _embed_text(query)
            if q_emb is None:
                return self._keyword_retrieve(query, top_k)
            scored = []
            for item in self.items:
                if item.get("embedding"):
                    score = self._cosine(q_emb, item["embedding"])
                else:
                    score = self._keyword_score(query, item["query"] + " " + item["answer"])
                scored.append((score, item))
        else:
            return self._keyword_retrieve(query, top_k)

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"score": round(s, 3), "query": it["query"], "answer": it["answer"]}
            for s, it in scored[:top_k] if s > 0.3
        ]

    def _keyword_retrieve(self, query: str, top_k: int) -> list[dict]:
        scored = []
        for item in self.items:
            score = self._keyword_score(query, item["query"] + " " + item["answer"])
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"score": round(s, 3), "query": it["query"], "answer": it["answer"]}
            for s, it in scored[:top_k] if s > 0.05
        ]

    # ---------- 持久化 ----------
    def save(self, force: bool = False):
        """写盘。force=True 忽略 dirty 检查"""
        if not force and not self._dirty:
            return
        data = {"id_counter": self._id_counter, "items": self.items}
        self.storage_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        self._dirty = False
        self._add_since_save = 0
        self._last_save_time = time.time()

    def load(self):
        if self.storage_path.is_file():
            try:
                data = json.loads(self.storage_path.read_text(encoding="utf-8"))
                self._id_counter = data.get("id_counter", 0)
                self.items = data.get("items", [])
                self._has_embedder = _get_embed_model() is not None
                self._migrate_embeddings()
                print(f"[RAG] 已加载 {len(self.items)} 条记忆"
                      f"(embedding: {'✓' if self._has_embedder else '✗ 关键词回退'})")
            except Exception as e:
                print(f"[RAG] 加载记忆文件失败: {e}")
                self.items = []
        else:
            self._has_embedder = _get_embed_model() is not None

    def _migrate_embeddings(self):
        """给加载的旧数据中没有 embedding 的条目补上向量"""
        if not self._has_embedder:
            return
        need_migrate = [it for it in self.items if it.get("embedding") is None]
        if not need_migrate:
            return
        print(f"[RAG] 正在为 {len(need_migrate)} 条旧记忆生成 embedding...")
        for it in need_migrate:
            text = f"问: {it['query']}\n答: {it['answer']}"
            it["embedding"] = _embed_text(text)
        self.save(force=True)
        print("[RAG] 迁移完成")

    def _on_exit(self):
        """进程退出时兜底写盘"""
        if self._dirty:
            self.save(force=True)
            print(f"[RAG] 退出时保存 {len(self.items)} 条记忆"
                  f"(已跳过 {self._dedup_skipped} 条重复)")


# -------------------- 记忆实例 --------------------
memory = RAGMemory(".agent_memory.json")
tools = CodeTools(root, memory=memory)


# -------------------- Prompt 模板 --------------------
PROMPT_TPL = """你是代码工程智能体（复刻Claude Code），负责分析、修改、调试本地项目。
你是一位陪伴用户写代码的 AI 助手，能记住过往对话，在需要时主动查阅或记录。

可用工具：
1. ls|路径 → 列出目录
2. read|文件相对路径 → 读取代码
3. write|文件路径|内容 → 写入/修改文件
4. cmd|命令字符串 → 执行终端命令
5. remember|查询关键词 → 搜索历史记忆（跨对话）
6. memorize|要记住的内容 → 手动存入长期记忆

输出格式严格二选一：
1. 需要操作工具：
Thought: 你的分析思考
Action: 工具名|参数1|参数2...

2. 任务完成，无需继续操作：
FinalAnswer: 总结做了哪些修改、验证结果、最终结论

约束：
- 修改代码前先读原文件，不要凭空改写
- 出现报错自动运行命令排查、修复代码再验证
- 不要一次性修改大量文件，小步迭代
- 不要超出项目根目录访问文件
- 如果当前问题与历史对话相关，先用 remember 工具搜索记忆
- 重要的设计决策、用户偏好、项目约定，用 memorize 工具存入记忆
- 任务完成后必须输出 FinalAnswer，不要反复执行相同操作

{memory}
上下文：{context}
用户需求：{user_query}"""


# -------------------- Agent 主逻辑 --------------------
def claude_code_agent(user_query: str, max_round=50):
    # 1. 检索相关历史记忆
    relevant = memory.retrieve(user_query, top_k=3)
    if relevant:
        memory_text = "【历史相关记忆】\n" + "\n".join(
            f"- [{m['score']:.2f}] 问: {m['query']}\n  答: {m['answer'][:300]}"
            for m in relevant
        )
    else:
        memory_text = ""

    context = ""
    round_idx = 0
    format_error_count = 0

    while round_idx < max_round:
        round_idx += 1
        print(f"\n===== 第{round_idx}轮思考 =====")

        extra_hint = ""
        if format_error_count >= 3:
            extra_hint = "\n【系统提示】你已经连续多次格式错误。请立即用 FinalAnswer 总结当前进展并结束本轮。"

        resp = client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {"role": "user", "content": PROMPT_TPL.format(
                    memory=memory_text,
                    context=context + extra_hint,
                    user_query=user_query
                )}
            ]
        )
        reply = resp.choices[0].message.content.strip()
        print(f"模型输出:\n{reply}")

        # 任务结束
        cleaned = reply.lstrip()
        if cleaned.startswith("FinalAnswer") and ":" in cleaned.split("\n")[0]:
            colon_idx = cleaned.index(":")
            result = cleaned[colon_idx + 1:].strip()
            memory.add(user_query, result)
            return result

        # 解析 Action
        lines = reply.splitlines()
        action_line = None
        thought = ""
        for line in lines:
            if line.startswith("Thought:"):
                thought = line
            if line.startswith("Action:"):
                action_line = line

        if not action_line:
            format_error_count += 1
            context += f"\n{reply}\nObservation: 格式错误，未识别有效指令（第{format_error_count}次）"
            continue

        format_error_count = 0

        action_content = action_line.replace("Action:", "").strip()
        parts = action_content.split("|")
        func_name = parts[0]
        args = parts[1:]

        if func_name not in TOOL_MAP:
            obs = f"错误：不存在工具 {func_name}"
        else:
            func = TOOL_MAP[func_name]
            obs = func(tools, *args)

        print(f"\n工具执行结果:\n{obs}")
        context += f"\n{thought}\n{action_line}\nObservation:\n{obs}"

    fallback = f"达到最大迭代轮次{max_round}，任务未完成。最后上下文：{context[-500:]}"
    memory.add(user_query, fallback)
    return fallback


# -------------------- 交互入口 --------------------
if __name__ == "__main__":
    print("=== 简易 Claude Code 复刻版（DeepSeek 驱动）===")
    print(f"记忆文件: .agent_memory.json  |  已有 {len(memory.items)} 条记忆")
    print("输入 exit 或 quit 退出程序\n")
    while True:
        user_input = input(">>> 请输入你的编程需求：")
        if user_input.strip().lower() in ("exit", "quit"):
            print("已退出。下次启动会自动加载记忆。")
            break
        result = claude_code_agent(user_input)
        print("\n========== 任务完成总结 ==========")
        print(result)
        print("\n" + "=" * 50)
