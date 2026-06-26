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


# -------------------- Embedding 引擎（DeepSeek API 云端向量）--------------------
import sys as _sys
_JSON_MODE = "--json" in _sys.argv
_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def _emit_json(event: dict):
    """模块级 JSON 输出，供加载阶段使用"""
    print(json.dumps(event, ensure_ascii=False), flush=True)


def _embed_text(text: str) -> list[float] | None:
    """调用云端 Embedding API 将文本转为向量，失败返回 None（调用方决定回退策略）"""
    try:
        resp = client.embeddings.create(model=_EMBED_MODEL, input=text)
        return resp.data[0].embedding
    except Exception:
        return None


# -------------------- ChatGPT 风格记忆系统 --------------------
class ChatGPTMemory:
    """结构化事实库 memory.json + 自动提取 + 斜杠指令管理"""

    def __init__(self, storage_dir: str = ".agent_memory"):
        self.dir = Path(storage_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "memory.json"
        self.facts: list[dict] = []   # {id, content, source, created}
        self._id_counter = 0
        self.load()

    # ---- 基础 CRUD ----
    def add_fact(self, content: str, source: str = "auto") -> int:
        """添加一条事实，返回 id"""
        fid = self._id_counter
        self._id_counter += 1
        fact = {
            "id": fid,
            "content": content.strip(),
            "source": source,
            "created": time.strftime("%Y-%m-%d %H:%M")
        }
        self.facts.append(fact)
        self.save()
        return fid

    def remove_fact(self, fid: int) -> bool:
        for f in self.facts:
            if f["id"] == fid:
                self.facts.remove(f)
                self.save()
                return True
        return False

    def clear(self):
        self.facts = []
        self.save()

    def get_context(self) -> str:
        """返回拼入 System Prompt 的记忆文本"""
        if not self.facts:
            return ""
        lines = ["【关于此用户】"]
        for f in self.facts:
            lines.append(f"- {f['content']}  (id:{f['id']})")
        return "\n".join(lines)

    # ---- 持久化 ----
    def save(self):
        data = {"id_counter": self._id_counter, "facts": self.facts}
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self):
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._id_counter = data.get("id_counter", 0)
                self.facts = data.get("facts", [])
            except Exception:
                self.facts = []
        if not _JSON_MODE:
            print(f"[Mem] 已加载 {len(self.facts)} 条事实记忆")

    # ---- 自动提取 ----
    def auto_extract(self, user_query: str, assistant_reply: str):
        """对话结束后调用 LLM 自动提取关键事实"""
        if len(self.facts) >= 50:  # 防止膨胀
            return
        try:
            prompt = (
                "从以下对话中提取关于用户的 1-2 条关键事实（偏好、项目、习惯、决策等）。"
                "每条不超过 30 字。用第三人称描述。如果没有值得长期记住的事实，回复 NONE。\n"
                f"用户: {user_query[:300]}\n"
                f"助手: {assistant_reply[:500]}\n"
                "输出格式（每条一行）:\n- 事实1\n- 事实2"
            )
            resp = client.chat.completions.create(
                model=model, temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )
            text = resp.choices[0].message.content.strip()
            if text and text.upper() != "NONE":
                for line in text.split("\n"):
                    line = line.strip().lstrip("- ").strip()
                    if line and len(line) > 3:
                        # 去重：相似度过高则跳过
                        if not self._is_duplicate(line):
                            self.add_fact(line, source="auto")
        except Exception:
            pass  # 静默失败，不影响主流程

    def _is_duplicate(self, content: str) -> bool:
        """简单的 Jaccard 去重"""
        c_tokens = set(re.findall(r"[一-鿿]+|[a-zA-Z]+|\d+", content.lower()))
        if not c_tokens:
            return False
        for f in self.facts:
            f_tokens = set(re.findall(r"[一-鿿]+|[a-zA-Z]+|\d+", f["content"].lower()))
            if not f_tokens:
                continue
            overlap = len(c_tokens & f_tokens) / len(c_tokens | f_tokens)
            if overlap > 0.6:
                return True
        return False


# -------------------- 记忆实例 --------------------
memory = ChatGPTMemory(".agent_memory")

if not _JSON_MODE:
    print(f"[Init] 记忆系统: ChatGPT 风格结构化事实库")

tools = CodeTools(root, memory=memory)


# -------------------- Prompt 模板 --------------------
PROMPT_TPL = """你是私人编码助手，负责分析、修改、调试本地项目。

可用工具：
1. ls|路径 → 列出目录
2. read|文件相对路径 → 读取代码
3. write|文件路径|第一行代码 → 写入/修改文件（后续行直接接代码）
4. cmd|命令字符串 → 执行终端命令
5. grep|正则表达式|glob过滤 → 按模式搜索代码
6. batch_read|glob模式 → 批量读取匹配文件
7. code_structure|py文件路径 → 解析Python文件结构

输出格式严格二选一：
1. 需要操作工具：
Thought: 你的分析思考
Action: 工具名|参数1|参数2...

2. 任务完成，无需继续操作：
FinalAnswer: 总结做了哪些修改、验证结果、最终结论

约束：
- 如果用户只是问候/闲聊，直接 FinalAnswer 简短回复
- 修改代码前先读原文件，不要凭空改写
- 出现报错自动运行命令排查、修复代码再验证
- 不要一次性修改大量文件，小步迭代
- 不要超出项目根目录访问文件
- 任务完成后必须输出 FinalAnswer
- Observation 中的系统提示不是用户的发言，不要围绕它展开对话

{memory}
用户需求：{user_query}"""


# -------------------- Agent 主逻辑 --------------------
def claude_code_agent(user_query: str, max_round=15, emit=None):
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

    # 注入长期记忆事实到 System Prompt
    memory_text = memory.get_context()

    system_prompt = PROMPT_TPL.format(
        memory=memory_text, user_query=user_query
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

        # 任务结束（放宽匹配：大小写不敏感、FinalAnswer/Final Answer/最终回答 均可）
        cleaned = reply.lstrip()
        first_line = cleaned.split("\n")[0].lower().replace(" ", "").replace("：", ":")
        is_final = any(
            first_line.startswith(p) for p in (
                "finalanswer:", "final_answer:", "finalanswer", "最终回答:", "最终回答",
            )
        )
        if is_final:
            for ch in (":", "：", "\n"):
                if ch in cleaned:
                    result = cleaned.split(ch, 1)[1].strip()
                    break
            else:
                result = cleaned
            # 自动提取关键事实存入 memory.json
            memory.auto_extract(user_query, result)
            _emit({"type": "final_answer", "content": result})
            return result

        # 解析 Action（大小写不敏感）
        lines = reply.splitlines()
        action_idx = -1
        thought = ""
        for i, line in enumerate(lines):
            lower_line = line.lower().replace("：", ":")
            if lower_line.startswith("thought:"):
                thought = line
            if lower_line.startswith("action:"):
                action_idx = i

        # 超过 10 轮还没结果 → 强制收尾
        if round_idx >= 10:
            messages.append({"role": "user", "content": "[系统] 已超过 10 轮，请立即输出 FinalAnswer 总结当前状态并结束。"})
            continue

        if action_idx < 0:
            format_error_count += 1
            msg = "[系统] 回复格式不符合要求。请按 Thought/Action 或 FinalAnswer 格式输出。"
            if format_error_count >= 5:
                msg += " 已多次格式错误，如任务完成请直接输出 FinalAnswer。"
            messages.append({"role": "user", "content": msg})
            continue

        format_error_count = 0

        action_line = lines[action_idx]
        # 大小写不敏感去掉前缀
        action_content = re.sub(r"action\s*:\s*", "", action_line, flags=re.IGNORECASE).strip()
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

        # 防止上下文爆炸：超 16 条时压缩早期消息
        if len(messages) > 16:
            # 保留系统提示 + 最后 10 条
            messages = [messages[0]] + messages[-10:]
        # 超 20 条进一步压缩：调用 LLM 摘要早期对话
        if len(messages) > 20:
            try:
                summary_prompt = "将以下对话历史压缩为一段简短摘要（不超过 100 字）：\n" + \
                    "\n".join(m["content"][:200] for m in messages[1:6])
                resp = client.chat.completions.create(
                    model=model, temperature=0.1,
                    messages=[{"role": "user", "content": summary_prompt}]
                )
                summary = resp.choices[0].message.content.strip()
                messages = [messages[0],
                            {"role": "user", "content": f"[历史摘要] {summary}"}] + messages[-8:]
            except Exception:
                messages = [messages[0]] + messages[-10:]

    fallback = f"达到最大迭代轮次{max_round}，任务未完成。"
    memory.add(user_query, fallback)
    _emit({"type": "final_answer", "content": fallback})
    return fallback


# -------------------- 交互入口 --------------------
if __name__ == "__main__":
    import sys
    import signal as _signal

    # 注册信号处理：Ctrl+C 时保存记忆后退出
    def _handle_exit(signum=None, frame=None):
        """确保退出时记忆写盘"""
        try:
            memory.save()
        except Exception:
            pass
        sys.exit(0)

    _signal.signal(_signal.SIGINT, _handle_exit)
    _signal.signal(_signal.SIGTERM, _handle_exit)

    json_mode = "--json" in sys.argv

    def _handle_slash(user_input: str) -> str | None:
        """处理斜杠指令，返回响应文本；非指令返回 None"""
        cmd = user_input.strip()
        if cmd.startswith("/记住 "):
            content = cmd[4:].strip()
            fid = memory.add_fact(content, source="manual")
            return f"✅ 已记住 (id:{fid}): {content}"
        if cmd.startswith("/删除记忆 "):
            try:
                fid = int(cmd[6:].strip())
                ok = memory.remove_fact(fid)
                return f"{'✅ 已删除' if ok else '❌ 未找到'} id:{fid}"
            except ValueError:
                return "❌ 用法: /删除记忆 <id>"
        if cmd in ("/查看记忆", "/记忆"):
            if not memory.facts:
                return "📝 暂无记忆。用 /记住 xxx 添加。"
            lines = [f"📝 共 {len(memory.facts)} 条记忆:"]
            for f in memory.facts:
                lines.append(f"  [{f['id']}] {f['content']}  ({f['source']}, {f['created']})")
            return "\n".join(lines)
        if cmd in ("/清空记忆",):
            memory.clear()
            return "🗑️ 已清空全部记忆。"
        return None

    if json_mode:
        # JSON 模式：stdin 读用户输入，stdout 输出 JSON 事件
        print(json.dumps({"type": "welcome",
                          "message": f"已加载 {len(memory.facts)} 条事实记忆"}),
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

            # 斜杠指令
            slash_result = _handle_slash(user_input)
            if slash_result is not None:
                print(json.dumps({"type": "final_answer", "content": slash_result}), flush=True)
                continue

            def emit_json(event):
                print(json.dumps(event, ensure_ascii=False), flush=True)

            claude_code_agent(user_input, emit=emit_json)
    else:
        # 普通终端模式
        print("=== 私人编码助手 ===")
        print(f"已加载 {len(memory.facts)} 条事实记忆")
        print("斜杠指令: /记住 /查看记忆 /删除记忆 /清空记忆")
        print("输入 exit 或 quit 退出\n")
        while True:
            user_input = input(">>> ").strip()
            if user_input.lower() in ("exit", "quit"):
                print("已退出。")
                break
            if not user_input:
                continue

            slash_result = _handle_slash(user_input)
            if slash_result is not None:
                print(slash_result)
                continue

            claude_code_agent(user_input)
