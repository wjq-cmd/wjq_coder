import { spawn } from "node:child_process";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import chalk from "chalk";
const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..", "..");
const PY_SCRIPT = join(ROOT, "claude_code_clone.py");
// ========== 调色板（对齐官方 Claude Code 配色） ==========
const C = {
    brand: chalk.hex("#D97757"), // claude orange
    thinking: chalk.hex("#8B9DC3"), // thinking blue-gray
    tool: chalk.hex("#E6C76A"), // tool call amber
    success: chalk.hex("#7EC87B"), // green
    error: chalk.hex("#E05555"), // red
    warning: chalk.hex("#E6C76A"), // yellow
    muted: chalk.hex("#999999"), // dim gray
    subtle: chalk.hex("#666666"), // very dim
    accent: chalk.hex("#C678DD"), // purple
    inverse: chalk.bgHex("#D97757").hex("#1a1a2e"), // inverse brand
};
// ========== 工具函数 ==========
const W = Math.min(process.stdout.columns || 100, 100);
function wrap(text, max = W - 4) {
    const out = [];
    for (const para of text.split("\n")) {
        if (para.length <= max) {
            out.push(para);
            continue;
        }
        let r = para;
        while (r.length > max) {
            out.push(r.slice(0, max));
            r = r.slice(max);
        }
        if (r)
            out.push(r);
    }
    return out.join("\n");
}
function divider(label) {
    if (label) {
        const side = "─".repeat(Math.max(2, (W - label.length - 4) / 2));
        console.log(C.subtle(`${side} ${label} ${side}`));
    }
    else {
        console.log(C.subtle("─".repeat(W)));
    }
}
/** 截断长文本 */
function truncate(text, max) {
    return text.length > max ? text.slice(0, max - 3) + "..." : text;
}
// ========== 工具元数据 ==========
const TOOL_META = {
    ls: { icon: "📂", label: "列出", color: C.tool },
    read: { icon: "📄", label: "读取", color: C.tool },
    write: { icon: "✎ ", label: "写入", color: C.accent },
    cmd: { icon: "⚡", label: "执行", color: C.warning },
    grep: { icon: "⌕ ", label: "搜索", color: C.tool },
    batch_read: { icon: "📚", label: "批量读取", color: C.tool },
    code_structure: { icon: "▥ ", label: "解析结构", color: C.tool },
    remember: { icon: "◷ ", label: "记忆搜索", color: C.brand },
    memorize: { icon: "✚ ", label: "存入记忆", color: C.brand },
};
// ========== 自定义 Spinner 帧 ==========
const SPINNER_FRAMES = ["◌", "◍", "◉", "◎", "○", "◎", "◉", "◍"];
let spinnerTimer = null;
let spinnerIdx = 0;
let spinnerText = "";
let spinnerActive = false;
function spinnerStart(text) {
    spinnerStop();
    spinnerText = text;
    spinnerActive = true;
    spinnerIdx = 0;
    renderSpinner();
    spinnerTimer = setInterval(() => {
        spinnerIdx = (spinnerIdx + 1) % SPINNER_FRAMES.length;
        renderSpinner();
    }, 80);
}
function spinnerUpdate(text) {
    spinnerText = text;
}
function spinnerStop(final) {
    if (spinnerTimer) {
        clearInterval(spinnerTimer);
        spinnerTimer = null;
    }
    if (spinnerActive) {
        process.stdout.write("\r\x1b[K"); // 清除当前行
        if (final)
            console.log(C.muted(`  ${final}`));
    }
    spinnerActive = false;
}
function renderSpinner() {
    if (!spinnerActive)
        return;
    const frame = SPINNER_FRAMES[spinnerIdx];
    process.stdout.write(`\r\x1b[K  ${C.brand(frame)} ${C.muted(spinnerText)}`);
}
// ========== 渲染函数 ==========
/** 模型思考内容 */
function renderLLMOutput(content) {
    const lines = content.split("\n");
    let hasThought = false;
    for (const line of lines) {
        if (line.startsWith("Thought:")) {
            const text = line.replace("Thought:", "").trim();
            if (text) {
                console.log(C.thinking(`  💭 ${text}`));
                hasThought = true;
            }
        }
        else if (line.startsWith("Action:")) {
            const text = line.replace("Action:", "").trim();
            const parts = text.split("|");
            const tool = parts[0];
            const meta = TOOL_META[tool] ?? { icon: "🔧", label: tool, color: C.tool };
            const arg1 = parts.length > 1 ? truncate(parts[1], 50) : "";
            const detail = arg1 ? C.muted(` → ${arg1}`) : "";
            console.log(`  ${meta.color(meta.icon)} ${chalk.bold(meta.color(meta.label))}${detail}`);
        }
        else if (line.trim()) {
            // 代码内容行（如 write 的代码）
            console.log(C.subtle(`  │ ${line.slice(0, 80)}`));
        }
    }
}
/** 工具执行结果（⎿ 缩进风格，对齐官方） */
function renderToolResult(tool, result) {
    const meta = TOOL_META[tool] ?? { icon: "🔧", label: tool, color: C.tool };
    const lines = result.trim().split("\n");
    const maxShow = 20;
    const display = lines.slice(0, maxShow);
    const truncated = lines.length > maxShow;
    // 第一行带 ⎿ 前缀
    if (display.length > 0) {
        console.log(C.muted(`  ⎿  ${meta.color(display[0])}`));
        for (let i = 1; i < display.length; i++) {
            console.log(C.muted(`     ${meta.color(display[i])}`));
        }
    }
    if (truncated) {
        console.log(C.subtle(`     ... 共 ${lines.length} 行`));
    }
}
/** 工具状态指示器（◌ 开始 / ○ 完成） */
function renderToolStatus(tool, done) {
    const meta = TOOL_META[tool] ?? { icon: "🔧", label: tool, color: C.tool };
    const dot = done ? C.success("○") : C.brand("◉");
    // 工具状态渲染在 tool_result 里，这里只做状态提示
}
/** 最终回答 */
function renderFinalAnswer(content) {
    divider("结果");
    console.log(C.success(wrap(content)));
    divider();
}
/** 轮次标题 */
function renderRound(n) {
    if (n > 1)
        console.log(""); // 轮次间空行
    console.log(C.subtle(`  ── 第 ${n} 轮 ──`));
}
// ========== 主流程 ==========
async function main() {
    console.clear();
    // 欢迎横幅
    console.log("");
    console.log(C.brand.bold("  ╔══════════════════════════════╗"));
    console.log(C.brand.bold("  ║       私 人 编 码 助 手       ║"));
    console.log(C.brand.bold("  ╚══════════════════════════════╝"));
    console.log("");
    console.log(C.subtle("  exit / quit 退出  |  输入需求开始对话"));
    console.log("");
    // ---- 启动 Python 持久进程 ----
    const proc = spawn("python", [PY_SCRIPT, "--json"], {
        cwd: ROOT,
        stdio: ["pipe", "pipe", "pipe"],
    });
    // SIGINT 处理：通知 Python 退出后再关自身
    process.on("SIGINT", () => {
        spinnerStop();
        proc.stdin.write("__EXIT__\n");
        setTimeout(() => process.exit(0), 300); // 给 Python 时间写盘
    });
    let currentResolve = null;
    let toolInProgress = null;
    // 读 stdout JSON 事件
    let buf = "";
    proc.stdout.on("data", (chunk) => {
        buf += chunk.toString();
        const lines = buf.split("\n");
        buf = lines.pop();
        for (const line of lines) {
            if (!line.trim())
                continue;
            try {
                handle(JSON.parse(line));
            }
            catch {
                console.log(C.muted(line));
            }
        }
    });
    // stderr 转发
    proc.stderr.on("data", (chunk) => {
        process.stderr.write(C.subtle(chunk.toString()));
    });
    proc.on("exit", (code) => {
        spinnerStop();
        if (currentResolve) {
            currentResolve();
            currentResolve = null;
        }
    });
    // ---- 事件分发 ----
    function handle(e) {
        switch (e.type) {
            // ---- 加载阶段 ----
            case "loading":
                if (e.status.includes("✓") || e.status.includes("就绪")) {
                    spinnerStop(`✔ ${e.status}`);
                }
                else {
                    spinnerStart(e.status);
                }
                break;
            case "welcome":
                spinnerStop();
                console.log(C.muted(`  ${e.message}`));
                break;
            // ---- Agent 循环 ----
            case "round":
                spinnerStop();
                if (toolInProgress) {
                    console.log(C.muted(`  ⎿  ...`)); // 上一工具结果收尾
                    toolInProgress = null;
                }
                renderRound(e.n);
                break;
            case "llm_start":
                spinnerStart("思考中...");
                break;
            case "llm_end":
                spinnerStop("✔");
                break;
            case "llm_output":
                renderLLMOutput(e.content);
                break;
            case "tool_result":
                renderToolResult(e.tool, e.result);
                toolInProgress = null;
                break;
            case "final_answer":
                spinnerStop();
                renderFinalAnswer(e.content);
                if (currentResolve) {
                    currentResolve();
                    currentResolve = null;
                }
                break;
            case "error":
                spinnerStop();
                console.log(C.error(`  ✗ ${e.message}`));
                break;
        }
    }
    // ---- 交互循环（原始 stdin）----
    process.stdin.setEncoding("utf8");
    process.stdin.resume();
    let inputBuf = "";
    let inputResolve = null;
    process.stdin.on("data", (chunk) => {
        inputBuf += chunk;
        if (inputBuf.includes("\n") && inputResolve) {
            const idx = inputBuf.indexOf("\n");
            const line = inputBuf.slice(0, idx).trim();
            inputBuf = inputBuf.slice(idx + 1);
            const resolve = inputResolve;
            inputResolve = null;
            resolve(line);
        }
    });
    function ask() {
        process.stdout.write(C.brand.bold("❯ "));
        return new Promise((r) => { inputResolve = r; });
    }
    while (true) {
        const input = await ask();
        if (!input)
            continue;
        if (["exit", "quit"].includes(input.toLowerCase())) {
            console.log(C.muted("\n  再见 👋\n"));
            proc.stdin.write("__EXIT__\n");
            break;
        }
        proc.stdin.write(input + "\n");
        await new Promise((r) => { currentResolve = r; });
        console.log("");
    }
    spinnerStop();
    process.stdin.pause();
}
main().catch(console.error);
