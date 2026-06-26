import os
import ast as py_ast
import re
import subprocess
import fnmatch
from pathlib import Path


def _safe_path(root: Path, rel_path: str) -> Path:
    """将相对路径解析到 root 下，阻止 ../ 逃逸"""
    p = (root / rel_path).resolve()
    if not str(p).startswith(str(root)):
        raise ValueError(f"路径越界: {rel_path}")
    return p


class CodeTools:
    def __init__(self, project_root: str, memory=None):
        self.root = Path(project_root).resolve()
        self.memory = memory
        self._file_cache: dict[str, str] = {}  # 路径 → 内容，会话级缓存

    def _cache_key(self, rel_path: str) -> str:
        return str((self.root / rel_path).resolve())

    # ========== 基础工具 ==========

    def ls_dir(self, rel_path: str = ".") -> str:
        """列出目录内容"""
        target = self.root / rel_path
        if not target.is_dir():
            return f"错误：{rel_path} 不是有效目录"
        items = sorted([i.name + ("/" if i.is_dir() else "") for i in target.iterdir()])
        return "\n".join(items) if items else "(空目录)"

    def read_file(self, rel_path: str) -> str:
        """读取单个文件（优先命中缓存）"""
        key = self._cache_key(rel_path)
        if key in self._file_cache:
            return self._file_cache[key]
        fp = self.root / rel_path
        if not fp.is_file():
            return f"文件不存在: {rel_path}"
        try:
            content = fp.read_text(encoding="utf-8")
            self._file_cache[key] = content
            return content
        except Exception as e:
            return f"读取失败: {str(e)}"

    def diff_preview(self, rel_path: str, new_content: str) -> str:
        """生成文件修改差异预览（优先用缓存里的旧版本）"""
        import difflib
        key = self._cache_key(rel_path)
        fp = self.root / rel_path
        # 缓存优先 → 磁盘兜底
        old = self._file_cache.get(key)
        if old is None:
            old = fp.read_text(encoding="utf-8") if fp.is_file() else ""
        old_lines = old.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        diff = difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{rel_path}", tofile=f"b/{rel_path}",
            lineterm=""
        )
        result = "\n".join(diff)
        return result if result else "(新文件，无差异)"

    def write_file(self, rel_path: str, content: str) -> str:
        """覆写/新建文件，同步更新缓存"""
        fp = self.root / rel_path
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        # 同步缓存：后续 read 直接拿最新版
        self._file_cache[self._cache_key(rel_path)] = content
        return f"已写入文件: {rel_path}"

    def run_command(self, cmd: str) -> str:
        """执行终端命令"""
        try:
            res = subprocess.run(
                cmd, shell=True, cwd=self.root,
                capture_output=True, text=True, timeout=60
            )
            return f"【stdout】\n{res.stdout}\n【stderr】\n{res.stderr}"
        except subprocess.TimeoutExpired:
            return "命令执行超时(60s)"
        except Exception as e:
            return f"命令执行异常: {str(e)}"

    # ========== 代码搜索 ==========

    def grep(self, pattern: str, glob_filter: str = "*") -> str:
        """
        按正则搜索代码内容，返回匹配行及上下文。
        用法: grep|pattern|glob_filter
        示例: grep|def login|**/*.py
        """
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"正则表达式错误: {e}"

        results = []
        # 收集匹配的文件
        files = []
        for fp in self.root.rglob(glob_filter):
            if fp.is_file() and fp.suffix in (".py", ".js", ".ts", ".tsx", ".jsx",
                                               ".html", ".css", ".json", ".yaml", ".yml",
                                               ".md", ".txt", ".toml", ".cfg", ".ini",
                                               ".sh", ".bat", ".sql", ".rs", ".go", ".java"):
                # 跳过常见的非代码目录
                parts = fp.parts
                if any(skip in parts for skip in (".git", "__pycache__", "node_modules",
                                                   ".venv", "venv", ".tox", ".mypy_cache",
                                                   "dist", "build", ".egg-info")):
                    continue
                files.append(fp)

        # 限制搜索范围，避免超时
        if len(files) > 500:
            files = files[:500]

        for fp in files:
            try:
                lines = fp.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue
            for i, line in enumerate(lines, 1):
                if regex.search(line):
                    rel = fp.relative_to(self.root)
                    results.append(f"{rel}:{i}: {line.strip()[:200]}")

        if not results:
            return f"未找到匹配 '{pattern}' 的内容"
        if len(results) > 50:
            return "\n".join(results[:50]) + f"\n... (共 {len(results)} 条，已截断至前50条)"
        return f"共 {len(results)} 条匹配:\n" + "\n".join(results)

    # ========== 批量读取 ==========

    def batch_read(self, glob_pattern: str) -> str:
        """
        批量读取匹配 glob 的文件内容。
        用法: batch_read|**/*.py
        每个文件带 --- 文件名 --- 分隔符。
        """
        files = sorted(self.root.glob(glob_pattern))
        files = [f for f in files if f.is_file()]
        # 跳过非代码目录
        files = [f for f in files if not any(
            skip in f.parts for skip in (
                ".git", "__pycache__", "node_modules", ".venv", "venv",
                "dist", "build", ".egg-info", ".mypy_cache"
            )
        )]

        if not files:
            return f"没有文件匹配: {glob_pattern}"
        if len(files) > 20:
            # 文件太多，只列出文件名
            rel_names = [str(f.relative_to(self.root)) for f in files]
            return f"匹配 {len(files)} 个文件（太多，用 read 单独读取）:\n" + "\n".join(rel_names[:50])

        parts = []
        total_chars = 0
        for fp in files:
            try:
                content = fp.read_text(encoding="utf-8")
            except Exception:
                continue
            rel = str(fp.relative_to(self.root))
            parts.append(f"--- {rel} ---\n{content}")
            total_chars += len(content)
            if total_chars > 30000:
                parts.append("... (总内容超过30000字符，已截断)")
                break

        return "\n\n".join(parts)

    # ========== 代码结构分析（Python） ==========

    def code_structure(self, rel_path: str) -> str:
        """
        解析 Python 文件的 AST 结构：类、函数、导入、顶层变量。
        用法: code_structure|相对路径
        """
        fp = self.root / rel_path
        if not fp.is_file():
            return f"文件不存在: {rel_path}"
        if not fp.suffix == ".py":
            return "目前仅支持解析 .py 文件"

        try:
            source = fp.read_text(encoding="utf-8")
            tree = py_ast.parse(source)
        except SyntaxError as e:
            return f"语法错误: {e}"
        except Exception as e:
            return f"解析失败: {e}"

        lines_out = []

        # 导入
        imports = []
        for node in py_ast.walk(tree):
            if isinstance(node, py_ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}")
            elif isinstance(node, py_ast.ImportFrom):
                names = ", ".join(a.name for a in node.names)
                imports.append(f"from {node.module or '.'} import {names}")
        if imports:
            lines_out.append("【导入】")
            lines_out.extend(f"  {imp}" for imp in imports)
            lines_out.append("")

        # 顶层定义：类、函数、赋值
        top_items = []
        for node in py_ast.iter_child_nodes(tree):
            info = self._node_info(node)
            if info:
                top_items.append(info)

        if top_items:
            lines_out.append("【顶层结构】")
            for item in top_items:
                lines_out.append(f"  {item}")

        return "\n".join(lines_out) if lines_out else "(文件为空或仅含表达式)"

    def _node_info(self, node, indent: int = 0) -> str | None:
        """递归提取节点信息"""
        prefix = "  " * indent

        if isinstance(node, py_ast.FunctionDef):
            sig = f"{node.name}("
            args = []
            for arg in node.args.args:
                a = arg.arg
                if arg.annotation:
                    a += f": {py_ast.unparse(arg.annotation)}"
                args.append(a)
            sig += ", ".join(args) + ")"
            if node.returns:
                sig += f" -> {py_ast.unparse(node.returns)}"
            # 装饰器
            decs = ""
            if node.decorator_list:
                decs = "[" + ", ".join(f"@{py_ast.unparse(d)}" for d in node.decorator_list) + "] "
            return f"{prefix}{decs}def {sig}  (行{node.lineno})"

        if isinstance(node, py_ast.AsyncFunctionDef):
            return f"{prefix}async def {node.name}(...)  (行{node.lineno})"

        if isinstance(node, py_ast.ClassDef):
            bases = ""
            if node.bases:
                bases = "(" + ", ".join(py_ast.unparse(b) for b in node.bases) + ")"
            decs = ""
            if node.decorator_list:
                decs = "[" + ", ".join(f"@{py_ast.unparse(d)}" for d in node.decorator_list) + "] "
            body = []
            for child in node.body:
                info = self._node_info(child, indent + 1)
                if info:
                    body.append(info)
            lines = [f"{prefix}{decs}class {node.name}{bases}  (行{node.lineno})"]
            lines.extend(body)
            return "\n".join(lines)

        if isinstance(node, py_ast.Assign):
            targets = ", ".join(py_ast.unparse(t) for t in node.targets)
            val = py_ast.unparse(node.value)
            if len(val) > 80:
                val = val[:80] + "..."
            return f"{prefix}{targets} = {val}  (行{node.lineno})"

        if isinstance(node, py_ast.AnnAssign):
            target = py_ast.unparse(node.target)
            ann = py_ast.unparse(node.annotation) if node.annotation else "?"
            return f"{prefix}{target}: {ann}  (行{node.lineno})"

        return None

    # ========== 记忆工具 ==========

    def memory_search(self, query: str) -> str:
        """搜索历史记忆"""
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


# ========== 工具映射 ==========
TOOL_MAP = {
    "ls":             CodeTools.ls_dir,
    "read":           CodeTools.read_file,
    "write":          CodeTools.write_file,
    "cmd":            CodeTools.run_command,
    "grep":           CodeTools.grep,
    "batch_read":     CodeTools.batch_read,
    "code_structure": CodeTools.code_structure,
}
