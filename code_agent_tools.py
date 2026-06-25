import os
import subprocess
from pathlib import Path


class CodeTools:
    def __init__(self, project_root: str, memory=None):
        self.root = Path(project_root).resolve()
        self.memory = memory  # RAGMemory 实例引用

    def ls_dir(self, rel_path: str = ".") -> str:
        """列出目录内容，对应claude查看目录"""
        target = self.root / rel_path
        if not target.is_dir():
            return f"错误：{rel_path} 不是有效目录"
        items = sorted([i.name + ("/" if i.is_dir() else "") for i in target.iterdir()])
        return "\n".join(items)

    def read_file(self, rel_path: str) -> str:
        """读取单个文件"""
        fp = self.root / rel_path
        if not fp.is_file():
            return f"文件不存在: {rel_path}"
        try:
            return fp.read_text(encoding="utf-8")
        except Exception as e:
            return f"读取失败: {str(e)}"

    def write_file(self, rel_path: str, content: str) -> str:
        """覆写/新建文件"""
        fp = self.root / rel_path
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return f"已写入文件: {rel_path}"

    def run_command(self, cmd: str) -> str:
        """执行终端命令，捕获输出（安全极简版）"""
        try:
            res = subprocess.run(
                cmd, shell=True, cwd=self.root,
                capture_output=True, text=True, timeout=60
            )
            out = res.stdout
            err = res.stderr
            return f"【stdout】\n{out}\n【stderr】\n{err}"
        except Exception as e:
            return f"命令执行异常: {str(e)}"

    def memory_search(self, query: str) -> str:
        """搜索历史记忆，找到与query最相关的过往对话"""
        if self.memory is None:
            return "记忆系统未启用"
        results = self.memory.retrieve(query, top_k=3)
        if not results:
            return "未找到相关历史记忆"
        lines = []
        for i, mem in enumerate(results, 1):
            lines.append(f"--- 记忆{i} (相似度:{mem['score']:.2f}) ---")
            lines.append(f"问: {mem['query']}")
            lines.append(f"答: {mem['answer'][:500]}")
        return "\n".join(lines)

    def memory_add(self, content: str) -> str:
        """手动添加一条长期记忆"""
        if self.memory is None:
            return "记忆系统未启用"
        self.memory.add(content, "(手动记录)")
        return f"已存入记忆: {content[:200]}"


# 工具映射，字符串指令映射函数
TOOL_MAP = {
    "ls":        CodeTools.ls_dir,
    "read":      CodeTools.read_file,
    "write":     CodeTools.write_file,
    "cmd":       CodeTools.run_command,
    "remember":  CodeTools.memory_search,
    "memorize":  CodeTools.memory_add,
}
