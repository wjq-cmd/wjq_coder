"""
离线构建记忆向量缓存。
读取 .agent_memory/memory.md，用 DeepSeek Embedding API 为每条对话生成 embedding，
写入 .agent_memory/memory.vectors.json。

用法：
  python build_embeddings.py                # 默认路径
  python build_embeddings.py --dir ./my_mem # 自定义目录
  python build_embeddings.py --force        # 强制重建全部向量
"""
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
_embed_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[一-鿿]+|[a-zA-Z]+|\d+", text.lower())
    return [t for t in tokens if len(t) > 1]


def parse_memory_md(md_path: Path) -> list[dict]:
    """从 memory.md 解析对话条目，返回 [{id, query, answer, text}, ...]"""
    if not md_path.is_file():
        print(f"[错误] 找不到 {md_path}，请先运行一次编码助手生成对话记忆。")
        sys.exit(1)

    text = md_path.read_text(encoding="utf-8")
    blocks = re.split(r"\n---\n", text)

    items = []
    for idx, block in enumerate(blocks[1:]):  # 跳过文件头
        block = block.strip()
        if not block:
            continue
        qm = re.search(r"^###\s*(.+)", block, re.MULTILINE)
        query = qm.group(1).strip() if qm else ""
        answer = re.sub(r"^###\s*.+\n", "", block, count=1)
        answer = re.sub(r"^\*.*\*\s*\n", "", answer, count=1).strip()
        if not query and not answer:
            continue

        combined = f"问: {query}\n答: {answer}"
        items.append({
            "id": idx,
            "query": query,
            "answer": answer,
            "text": combined,
        })

    return items


def main():
    import argparse
    parser = argparse.ArgumentParser(description="构建记忆向量缓存")
    parser.add_argument("--dir", default=".agent_memory", help="记忆目录")
    parser.add_argument("--force", action="store_true", help="强制重建全部向量")
    args = parser.parse_args()

    mem_dir = Path(args.dir)
    md_path = mem_dir / "memory.md"
    vec_path = mem_dir / "memory.vectors.json"

    # 1) 解析 markdown
    items = parse_memory_md(md_path)
    print(f"从 {md_path} 解析出 {len(items)} 条对话")

    # 2) 读已有向量缓存（增量模式）
    existing = {}
    if vec_path.is_file() and not args.force:
        try:
            existing = json.loads(vec_path.read_text(encoding="utf-8")).get("embeddings", {})
            print(f"已有 {len(existing)} 条向量缓存，将只处理新增条目")
        except Exception:
            pass

    # 3) 找出需要生成向量的条目
    to_embed = []
    for item in items:
        eid = str(item["id"])
        if eid not in existing or args.force:
            to_embed.append(item)

    if not to_embed:
        print("所有条目已有向量缓存，无需重建。")
        return

    print(f"需要为 {len(to_embed)} 条对话生成向量...")

    # 4) 调用云端 Embedding API 逐条编码
    total = len(to_embed)
    print(f"正在通过云端 API 编码 {total} 条文本 (model={_embed_model})...")
    start = time.time()

    for i, item in enumerate(to_embed):
        try:
            resp = _client.embeddings.create(model=_embed_model, input=item["text"])
            existing[str(item["id"])] = resp.data[0].embedding
        except Exception as e:
            print(f"  [{i+1}/{total}] 编码失败: {e}")
            continue
        if (i + 1) % 10 == 0 or i == total - 1:
            elapsed = time.time() - start
            print(f"  [{i+1}/{total}] 进度，耗时 {elapsed:.1f}s")

    elapsed = time.time() - start
    ok = sum(1 for it in to_embed if str(it["id"]) in existing)
    print(f"编码完成: {ok}/{total} 条成功，耗时 {elapsed:.1f}s")

    # 6) 写入
    dim = len(next(iter(existing.values()))) if existing else 0
    data = {
        "model": _embed_model,
        "dim": dim,
        "count": len(existing),
        "embeddings": existing,
    }
    vec_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"已写入 {vec_path} ({len(existing)} 条向量，{vec_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
