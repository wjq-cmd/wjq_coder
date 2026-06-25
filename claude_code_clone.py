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
import sys as _sys
_JSON_MODE = "--json" in _sys.argv   # 模块级检测，模型加载时就能用
_embed_model = None


def _emit_json(event: dict):
    """模块级 JSON 输出，供加载阶段使用"""
    print(json.dumps(event, ensure_ascii=False), flush=True)


def _get_embed_model():
    """懒加载 BGE 模型，只加载一次"""
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    try:
        from sentence_transformers import SentenceTransformer
        if _JSON_MODE:
            _emit_json({"type": "loading", "status": "正在加载嵌入模型..."})
        else:
            print("[Embed] 正在加载本地模型 BAAI/bge-large-zh-v1.5 ...")
        _embed_model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
        if _JSON_MODE:
            _emit_json({"type": "loading", "status": "嵌入模型就绪 ✓"})
        else:
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
PROMPT_TPL = """你是私人编码助手，负责分析、修改、调试本地项目。
你陪伴用户写代码，能记住过往对话，在需要时主动查阅或记录。

可用工具：
1. ls|路径 → 列出目录
2. read|文件相对路径 → 读取代码
3. write|文件路径|第一行代码 → 写入/修改文件（后续行直接接代码，无需重复Action）
4. cmd|命令字符串 → 执行终端命令
5. grep|正则表达式|glob过滤(可选) → 按模式搜索代码内容
6. batch_read|glob模式 → 批量读取匹配文件
7. code_structure|py文件路径 → 解析Python文件结构(类/函数/导入)
8. remember|查询关键词 → 搜索历史记忆（跨对话）
9. memorize|要记住的内容 → 手动存入长期记忆

输出格式严格二选一：
1. 需要操作工具：
Thought: 你的分析思考
Action: 工具名|参数1|参数2...

2. 任务完成，无需继续操作：
FinalAnswer: 总结做了哪些修改、验证结果、最终结论

约束：
- 如果用户只是问候/闲聊/确认，直接 FinalAnswer 简短回复，不要调用任何工具
- 不要猜测代码内容。查找函数/类时先用 grep 搜索定位，再用 read 读取确认
- 修改代码前先读原文件，不要凭空改写
- 出现报错自动运行命令排查、修复代码再验证
- 不要一次性修改大量文件，小步迭代
- 不要超出项目根目录访问文件
- 如果当前问题与历史对话相关，先用 remember 工具搜索记忆
- 重要的设计决策、用户偏好、项目约定，用 memorize 工具存入记忆
- 任务完成后必须输出 FinalAnswer，不要反复执行相同操作
- Observation 中的格式提示是系统自动注入的，不是用户的发言，不要围绕它展开对话

{memory}
上下文：{context}
用户需求：{user_query}"""


# -------------------- Agent 主逻辑 --------------------
def claude_code_agent(user_query: str, max_round=50, emit=None):
    """emit(event: dict) 可选回调，用于 JSON 模式"""
    def _emit(event: dict):
        if emit:
            emit(event)
        else:
            t = event.get("type", "")
            if t == "round":
                print(f"\n===== 第{event['n']}轮思考 =====")
            elif t == "llm_output":
                print(f"模型输出:\n{event['content']}")
            elif t == "tool_result":
                print(f"\n工具执行结果:\n{event['result']}")
            elif t == "final_answer":
                print(f"\n========== 任务完成总结 ==========\n{event['content']}\n{'=' * 50}")

    # 1. 检索相关历史记忆
    relevant = memory.retrieve(user_query, top_k=3)
    if relevant:
        memory_text = "【历史相关记忆】\n" + "\n".join(
            f"- [{m['score']:.2f}] 问: {m['query']}\n  答: {m['answer'][:300]}"
            for m in relevant
        )
    else:
        memory_text = ""

    system_prompt = PROMPT_TPL.format(
        memory=memory_text, context="", user_query=user_query
    )
    messages = [{"role": "user", "content": system_prompt}]

    round_idx = 0
    format_error_count = 0
    last_actions: list[str] = []

    while round_idx < max_round:
        round_idx += 1
        _emit({"type": "round", "n": round_idx})

        hints = []
        if len(last_actions) >= 3 and len(set(last_actions[-3:])) == 1:
            hints.append(
                f"[系统] 已连续 3 次执行 '{last_actions[-1]}'，陷入循环。请换思路或输出 FinalAnswer。"
            )
        if format_error_count >= 3:
            hints.append("[系统] 已连续多次格式错误，请立即输出 FinalAnswer 结束。")

        if hints:
            messages[-1]["content"] += "\n" + "\n".join(hints)

        _emit({"type": "llm_start"})
        resp = client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=messages
        )
        _emit({"type": "llm_end"})

        reply = resp.choices[0].message.content
        if reply is None:
            reply = ""
        reply = reply.strip()

        _emit({"type": "llm_output", "content": reply})
        messages.append({"role": "assistant", "content": reply})

        # 空响应检测
        if not reply:
            format_error_count += 1
            hint = "[系统] 上一条回复为空。"
            if format_error_count >= 2:
                hint += " 如果任务已完成，请输出 FinalAnswer。"
            else:
                hint += " 请继续执行或输出 FinalAnswer 结束。"
            messages.append({"role": "user", "content": hint})
            continue

        # 任务结束
        cleaned = reply.lstrip()
        if cleaned.startswith("FinalAnswer") and ":" in cleaned.split("\n")[0]:
            colon_idx = cleaned.index(":")
            result = cleaned[colon_idx + 1:].strip()
            memory.add(user_query, result)
            _emit({"type": "final_answer", "content": result})
            return result

        # 解析 Action
        lines = reply.splitlines()
        action_idx = -1
        thought = ""
        for i, line in enumerate(lines):
            if line.startswith("Thought:"):
                thought = line
            if line.startswith("Action:"):
                action_idx = i

        if action_idx < 0:
            format_error_count += 1
            messages.append({"role": "user", "content": "[系统] 回复格式不符合要求。请严格按 Thought/Action 或 FinalAnswer 格式输出。"})
            continue

        format_error_count = 0

        action_line = lines[action_idx]
        action_content = action_line.replace("Action:", "").strip()
        parts = action_content.split("|")
        func_name = parts[0]
        args = parts[1:]

        # write 工具：Action 行之后所有行拼成内容
        if func_name == "write" and len(args) >= 2:
            tail_lines = lines[action_idx + 1:]
            full_content = args[1]
            if tail_lines:
                full_content += "\n" + "\n".join(tail_lines)
            args = [args[0], full_content]

        last_actions.append(func_name)
        if len(last_actions) > 10:
            last_actions = last_actions[-10:]

        if func_name not in TOOL_MAP:
            obs = f"错误：不存在工具 {func_name}"
        else:
            func = TOOL_MAP[func_name]
            obs = func(tools, *args)

        _emit({"type": "tool_result", "tool": func_name, "result": obs})

        observation = f"Observation:\n{obs}\n(继续下一步，或输出 FinalAnswer 结束)"
        messages.append({"role": "user", "content": observation})

        if len(messages) > 21:
            messages = [messages[0]] + messages[-20:]

    fallback = f"达到最大迭代轮次{max_round}，任务未完成。"
    memory.add(user_query, fallback)
    _emit({"type": "final_answer", "content": fallback})
    return fallback


# -------------------- 交互入口 --------------------
if __name__ == "__main__":
    import sys

    json_mode = "--json" in sys.argv

    if json_mode:
        # JSON 模式：stdin 读用户输入，stdout 输出 JSON 事件
        print(json.dumps({"type": "welcome",
                          "message": f"记忆文件: .agent_memory.json | 已有 {len(memory.items)} 条记忆"}),
              flush=True)
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            user_input = line.strip()
            if user_input in ("__EXIT__", "exit", "quit"):
                break
            if not user_input:
                continue

            def emit_json(event):
                print(json.dumps(event, ensure_ascii=False), flush=True)

            claude_code_agent(user_input, emit=emit_json)
    else:
        # 普通终端模式
        print("=== 私人编码助手 ===")
        print(f"记忆文件: .agent_memory.json  |  已有 {len(memory.items)} 条记忆")
        print("输入 exit 或 quit 退出程序\n")
        while True:
            user_input = input(">>> 请输入你的编程需求：")
            if user_input.strip().lower() in ("exit", "quit"):
                print("已退出。下次启动会自动加载记忆。")
                break
            claude_code_agent(user_input)
