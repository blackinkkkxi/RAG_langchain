from __future__ import annotations

import argparse
import html
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request
from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.5-27b"
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

HELP_DETAILS = "\n".join(
    [
        "Commands:",
        "/help    Show this help message.",
        "/memory  Show working memory.",
        "/session Show the current session file path.",
        "/reset   Clear history and working memory.",
        "/exit    Exit the agent.",
    ]
)

MAX_TOOL_OUTPUT = 4000
MAX_HISTORY = 12000
MAX_STATUS_CHARS = 2000
MAX_MEMORY_CHARS = 40000
MAX_WEB_BYTES = 250_000
MAX_WEB_TEXT = 20_000
DEFAULT_WEB_TIMEOUT = 20

NETWORK_USER_AGENT = (
    "Mozilla/5.0 (compatible; mini-claude-code/1.0; +https://example.invalid)"
)

IGNORED_PATH_NAMES = {
    ".git",
    ".mini-coding-agent",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
}

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
MEMORY_INSTRUCTION_PROMPT = (
    "Codebase and user instructions are shown below. Be sure to adhere to these "
    "instructions. IMPORTANT: These instructions OVERRIDE any default behavior "
    "and you MUST follow them exactly as written."
)

def build_welcome(agent, model: str, endpoint: str) -> str:
    RESET = "\033[0m"
    BOLD = "\033[1m"

    # 颜色定义
    FG_TITLE = "\033[38;5;223m"  # 暖白
    FG_CAT = "\033[38;5;216m"    # 奶橘
    FG_LABEL = "\033[38;5;110m"  # 青灰
    FG_VALUE = "\033[38;5;252m"  # 浅灰
    FG_BOX = "\033[38;5;67m"     # 蓝灰
    FG_CMD = "\033[38;5;180m"    # 淡金

    cat_art = [
        f"{FG_CAT}    /\\_/\\      {RESET}",
        f"{FG_CAT}   ( o.o )     {RESET}",
        f"{FG_CAT}    > ^ <      {RESET}",
        f"{FG_CAT}   /     \\     {RESET}",
        f"{FG_CAT}   (___)__ )   {RESET}",
    ]

    info = [
        ("workspace", agent.workspace.cwd),
        ("model", model),
        ("endpoint", endpoint),
        ("branch", agent.workspace.branch),
        ("approval", agent.approval_policy),
        ("session", agent.session["id"]),
    ]
    def _plain(s: str) -> str:
        return re.sub(r"\033\[[0-9;]*m", "", s)

    cat_width = max(len(_plain(line)) for line in cat_art)
    label_width = max(len(k) for k, v in info)

    info_plain_lines = [f"{k}: {v}" for k, v in info]
    right_width = max(len(line) for line in info_plain_lines)

    gap = "  │  " 
    content_inner_width = cat_width + len(_plain(gap)) + right_width

    cmd_text = "commands: /help, /memory, /session, /reset, /exit"
    title_text = "MINI-CLAUDE-CODE"
    total_inner_width = max(content_inner_width, len(cmd_text), len(title_text)) + 4

    lines = []

    lines.append(f"{FG_BOX}╭{'─' * total_inner_width}╮{RESET}")

    title_pad = (total_inner_width - len(title_text)) // 2
    lines.append(f"{FG_BOX}│{RESET}{' ' * title_pad}{BOLD}{FG_TITLE}{title_text}{RESET}{' ' * (total_inner_width - len(title_text) - title_pad)}{FG_BOX}│{RESET}")

    lines.append(f"{FG_BOX}├{'─' * total_inner_width}┤{RESET}")

    max_rows = max(len(cat_art), len(info))
    for i in range(max_rows):
        left = cat_art[i] if i < len(cat_art) else " " * cat_width
        if i < len(info):
            k, v = info[i]
            k_str = f"{FG_LABEL}{k.ljust(label_width)}{RESET}"
            right = f"{k_str}: {FG_VALUE}{v}{RESET}"
            right_plain_len = label_width + 2 + len(str(v))
        else:
            right = ""
            right_plain_len = 0

        current_row_plain_width = cat_width + len(_plain(gap)) + right_plain_len
        padding = " " * (total_inner_width - current_row_plain_width - 2) # -2 是前后的空格

        lines.append(f"{FG_BOX}│{RESET} {left}{gap}{right}{padding} {FG_BOX}│{RESET}")
    lines.append(f"{FG_BOX}├{'─' * total_inner_width}┤{RESET}")
    cmd_padding = " " * (total_inner_width - len(cmd_text) - 2)
    lines.append(f"{FG_BOX}│{RESET} {FG_CMD}{cmd_text}{RESET}{cmd_padding} {FG_BOX}│{RESET}")
    lines.append(f"{FG_BOX}╰{'─' * total_inner_width}╯{RESET}")

    return "\n".join(lines)

class DashScopeModelClient:
    def __init__(
        self,
        model: str,
        api_key: str | None,
        base_url: str,
        temperature: float,
        top_p: float,
        timeout: int,
        enable_thinking: bool = True,
        client=None,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.enable_thinking = enable_thinking
        if client is not None:
            self.client = client
            return
        if OpenAI is None:
            raise RuntimeError(
                "The `openai` package is required. Install it with `pip install openai`."
            )
        if not self.api_key:
            raise RuntimeError(
                "DashScope API key is missing. Set `DASHSCOPE_API_KEY` or pass `--api-key`."
            )
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def complete(self, messages: list[dict[str, str]], max_new_tokens: int) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=max_new_tokens,
                extra_body={"enable_thinking": self.enable_thinking},
            )
        except Exception as exc:
            raise RuntimeError(
                "DashScope request failed.\n"
                "Confirm the API key, base URL, and model name.\n"
                f"Base URL: {self.base_url}\n"
                f"Model: {self.model}\n"
                f"Details: {exc}"
            ) from exc

        try:
            content = response.choices[0].message.content
        except Exception:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, str):
                    chunks.append(item)
                    continue
                text = getattr(item, "text", None)
                if text:
                    chunks.append(text)
            return "".join(chunks)
        return str(content or "")

def init_env() -> None:
    if load_dotenv is not None:
        load_dotenv(dotenv_path=ENV_FILE)

def clip(text: Any, limit: int = MAX_TOOL_OUTPUT) -> str:
    text = str(text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n...[truncated {len(text) - limit} chars]"

@dataclass
class MemoryFile:
    display_path: str
    path: str
    type: str
    content: str

##############################
#### 1) Live Repo Context ####
##############################
@dataclass
class WorkspaceContext:
    cwd: str
    repo_root: str
    branch: str
    default_branch: str
    status: str
    recent_commits: list[str]
    project_docs: dict[str, str]
    is_git_repository: bool = False
    environment_prompt: str = ""
    system_context: dict[str, str] = field(default_factory=dict)
    user_context: dict[str, str] = field(default_factory=dict)
    memory_files: list[MemoryFile] = field(default_factory=list)

    @classmethod
    def build(cls, cwd: str | os.PathLike[str]) -> "WorkspaceContext":
        cwd_path = Path(cwd).resolve()
        git_root = cls.find_git_root(cwd_path)
        canonical_root = cls.find_canonical_git_root(cwd_path)
        repo_root = (canonical_root or git_root or cwd_path).resolve()
        is_git_repository = git_root is not None

        branch = cls.detect_branch(cwd_path, is_git_repository)
        default_branch = cls.detect_default_branch(cwd_path, is_git_repository)
        git_user = cls.git(cwd_path, ["config", "user.name"], "") if is_git_repository else ""
        status = (
            cls.git(cwd_path, ["--no-optional-locks", "status", "--short"], "")
            if is_git_repository
            else ""
        )
        recent_commits = (
            [
                line
                for line in cls.git(
                    cwd_path,
                    ["--no-optional-locks", "log", "--oneline", "-n", "5"],
                    "",
                ).splitlines()
                if line.strip()
            ]
            if is_git_repository
            else []
        )

        # 仓库状态可能非常长；这里先在采集阶段截断，避免稳定上下文本身膨胀。
        if len(status) > MAX_STATUS_CHARS:
            status = (
                status[:MAX_STATUS_CHARS].rstrip()
                + '\n... (truncated because it exceeds 2k characters. '
                'If you need more information, run "git status" using BashTool)'
            )

        is_worktree = (
            git_root is not None
            and canonical_root is not None
            and git_root.resolve() != canonical_root.resolve()
        )

        environment_prompt = cls.build_environment_prompt(
            cwd=cwd_path,
            is_git_repository=is_git_repository,
            is_worktree=is_worktree,
        )
        memory_files = cls.collect_memory_files(cwd_path, repo_root, git_root, canonical_root)

        system_context: dict[str, str] = {}
        if is_git_repository:
            # git 快照在会话开始时采集一次，后续通过 append_system_context 反复复用。
            system_context["gitStatus"] = cls.build_git_snapshot(
                branch=branch,
                default_branch=default_branch,
                git_user=git_user,
                status=status or "(clean)",
                recent_commits=recent_commits,
            )

        user_context = {"currentDate": f"Today's date is {date.today().isoformat()}."}
        claude_md = cls.render_memory_prompt(memory_files)
        if claude_md:
            user_context["claudeMd"] = claude_md

        return cls(
            cwd=str(cwd_path),
            repo_root=str(repo_root),
            branch=branch,
            default_branch=default_branch,
            status=status or "(clean)",
            recent_commits=recent_commits,
            project_docs={item.display_path: item.content for item in memory_files},
            is_git_repository=is_git_repository,
            environment_prompt=environment_prompt,
            system_context=system_context,
            user_context=user_context,
            memory_files=memory_files,
        )

    @staticmethod
    def build_environment_prompt(
        *,
        cwd: Path,
        is_git_repository: bool,
        is_worktree: bool,
    ) -> str:
        items = [
            f"Primary working directory: {cwd}",
            (
                "This is a git worktree - an isolated copy of the repository. "
                "Run all commands from this directory. Do NOT `cd` to the original repository root."
                if is_worktree
                else None
            ),
            f"Is a git repository: {str(is_git_repository).lower()}",
            f"Platform: {os.name}/{platform.system().lower()}",
            WorkspaceContext.shell_info_line(),
            f"OS Version: {WorkspaceContext.os_version_line()}",
        ]
        return (
            "# Environment\n"
            "You have been invoked in the following environment:\n"
            f"{WorkspaceContext.prepend_bullets(items)}"
        )

    @staticmethod
    def build_git_snapshot(
        *,
        branch: str,
        default_branch: str,
        git_user: str,
        status: str,
        recent_commits: list[str],
    ) -> str:
        parts = [
            "This is the git status at the start of the conversation. "
            "Note that this status is a snapshot in time, and will not update during the conversation.",
            f"Current branch: {branch}",
            f"Main branch (you will usually use this for PRs): {default_branch}",
        ]
        if git_user:
            parts.append(f"Git user: {git_user}")
        parts.append(f"Status:\n{status}")
        parts.append(f"Recent commits:\n{chr(10).join(recent_commits) or '(none)'}")
        return "\n\n".join(parts)

    @classmethod
    def collect_memory_files(
        cls,
        cwd: Path,
        repo_root: Path,
        git_root: Path | None,
        canonical_root: Path | None,
    ) -> list[MemoryFile]:
        candidate_names = (
            "AGENTS.md",
            "AGENT.md",
            "CLAUDE.md",
            "agents.md",
            "agent.md",
            "claude.md",
        )

        current = cwd.resolve()
        while True:
            for name in candidate_names:
                path = current / name
                if not path.is_file():
                    continue
                raw = cls.read_text(path)
                if not raw:
                    continue
                content = clip(cls.strip_frontmatter(raw).strip(), MAX_MEMORY_CHARS)
                if not content:
                    continue
                return [
                    MemoryFile(
                        display_path=cls.doc_key(path, repo_root, cwd),
                        path=str(path.resolve()),
                        type="Instruction",
                        content=content,
                    )
                ]
            if current.parent == current:
                break
            current = current.parent

        return []

    @staticmethod
    def render_memory_prompt(memory_files: list[MemoryFile]) -> str:
        if not memory_files:
            return ""
        # 这批规则文本通常不会每轮变化，因此适合作为相对稳定的上下文切片复用。
        blocks = [
            f"Contents of {memory.path}{WorkspaceContext.memory_label(memory.type)}:\n\n{memory.content}"
            for memory in memory_files
        ]
        return MEMORY_INSTRUCTION_PROMPT + "\n\n" + "\n\n".join(blocks)

    def append_system_context(self, parts: list[str]) -> list[str]:
        # 在稳定前缀后追加 system 级上下文，避免每次从零拼整段大提示。
        tail = "\n".join(f"{key}: {value}" for key, value in self.system_context.items() if value)
        return [*parts, tail] if tail else list(parts)

    def user_context_message(self) -> str:
        if not self.user_context:
            return ""
        # 这里把较稳定的 user 上下文单独成段，便于和每轮更新的 transcript 分离。
        body = "\n".join(f"# {key}\n{value}" for key, value in self.user_context.items() if value)
        return (
            "<system-reminder>\n"
            "As you answer the user's questions, you can use the following context:\n"
            f"{body}\n\n"
            "IMPORTANT: this context may or may not be relevant to your tasks. "
            "You should not respond to this context unless it is highly relevant to your task.\n"
            "</system-reminder>\n"
        )

    def text(self) -> str:
        commits = "\n".join(f"- {item}" for item in self.recent_commits) or "- none"
        docs = "\n".join(f"- {path}\n{snippet}" for path, snippet in self.project_docs.items()) or "- none"
        return "\n".join(
            [
                "Workspace:",
                f"- cwd: {self.cwd}",
                f"- repo_root: {self.repo_root}",
                f"- is_git_repository: {str(self.is_git_repository).lower()}",
                f"- branch: {self.branch}",
                f"- default_branch: {self.default_branch}",
                "- status:",
                self.status,
                "- recent_commits:",
                commits,
                "- project_docs:",
                docs,
            ]
        )

    @staticmethod
    def run(argv: list[str], cwd: Path, timeout: int = 5) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    @staticmethod
    def git(cwd: Path, args: list[str], fallback: str = "") -> str:
        try:
            result = WorkspaceContext.run(["git", *args], cwd=cwd)
        except Exception:
            return fallback
        if result.returncode != 0:
            return fallback
        return result.stdout.strip() or fallback

    @staticmethod
    def git_ok(cwd: Path, args: list[str]) -> bool:
        try:
            return WorkspaceContext.run(["git", *args], cwd=cwd).returncode == 0
        except Exception:
            return False

    @staticmethod
    def is_relative_to(path: Path, base: Path) -> bool:
        try:
            path.resolve().relative_to(base.resolve())
            return True
        except Exception:
            return False

    @staticmethod
    def get_claude_config_home_dir() -> Path:
        return Path(os.environ.get("CLAUDE_CONFIG_DIR") or (Path.home() / ".claude")).resolve()

    @staticmethod
    def get_managed_file_path() -> Path:
        system = platform.system().lower()
        if system == "darwin":
            return Path("/Library/Application Support/ClaudeCode")
        if system == "windows":
            return Path(r"C:\Program Files\ClaudeCode")
        return Path("/etc/claude-code")

    @staticmethod
    def find_git_root(start: Path) -> Path | None:
        current = start.resolve()
        while True:
            if (current / ".git").exists():
                return current
            if current.parent == current:
                return None
            current = current.parent

    @staticmethod
    def resolve_git_dir(repo_root: Path) -> Path | None:
        marker = repo_root / ".git"
        if marker.is_dir():
            return marker.resolve()
        if not marker.is_file():
            return None
        try:
            content = marker.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            return None
        if content.startswith("gitdir:"):
            return (repo_root / content[len("gitdir:") :].strip()).resolve()
        return None

    @staticmethod
    def find_canonical_git_root(start: Path) -> Path | None:
        git_root = WorkspaceContext.find_git_root(start)
        if not git_root:
            return None
        git_dir = WorkspaceContext.resolve_git_dir(git_root)
        if not git_dir:
            return git_root
        commondir = git_dir / "commondir"
        if not commondir.exists():
            return git_root
        try:
            common_dir = (git_dir / commondir.read_text(encoding="utf-8").strip()).resolve()
        except Exception:
            return git_root
        return common_dir.parent if common_dir.name == ".git" else common_dir

    @staticmethod
    def detect_branch(cwd: Path, is_git_repository: bool) -> str:
        if not is_git_repository:
            return "HEAD"
        return WorkspaceContext.git(cwd, ["symbolic-ref", "--quiet", "--short", "HEAD"], "") or "HEAD"

    @staticmethod
    def detect_default_branch(cwd: Path, is_git_repository: bool) -> str:
        if not is_git_repository:
            return "main"
        origin_head = WorkspaceContext.git(
            cwd,
            ["symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD"],
            "",
        )
        if origin_head.startswith("origin/"):
            return origin_head.removeprefix("origin/")
        for candidate in ("main", "master"):
            if WorkspaceContext.git_ok(cwd, ["show-ref", "--verify", "--quiet", f"refs/remotes/origin/{candidate}"]):
                return candidate
        return "main"

    @staticmethod
    def shell_info_line() -> str:
        shell = os.environ.get("SHELL") or os.environ.get("COMSPEC") or "unknown"
        if "zsh" in shell:
            shell = "zsh"
        elif "bash" in shell:
            shell = "bash"
        if os.name == "nt":
            return (
                f"Shell: {shell} (use Unix shell syntax, not Windows syntax; "
                "prefer forward slashes and /dev/null-style paths)"
            )
        return f"Shell: {shell}"

    @staticmethod
    def os_version_line() -> str:
        return f"{platform.system()} {platform.release()}"

    @staticmethod
    def strip_frontmatter(text: str) -> str:
        match = FRONTMATTER_RE.match(text)
        return text[match.end() :] if match else text

    @staticmethod
    def has_conditional_paths_frontmatter(text: str) -> bool:
        match = FRONTMATTER_RE.match(text)
        if not match:
            return False
        return bool(re.search(r"(?m)^\s*paths\s*:", match.group(1)))

    @staticmethod
    def read_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    @staticmethod
    def doc_key(path: Path, repo_root: Path, cwd: Path) -> str:
        for base in (repo_root, cwd):
            try:
                return str(path.resolve().relative_to(base.resolve()))
            except Exception:
                pass
        return str(path.resolve())

    @staticmethod
    def prepend_bullets(items: list[Any]) -> str:
        lines: list[str] = []
        for item in items:
            if item is None:
                continue
            if isinstance(item, list):
                lines.extend(f"- {value}" for value in item if value)
            else:
                lines.append(f"- {item}")
        return "\n".join(lines)

    @staticmethod
    def memory_label(kind: str) -> str:
        labels = {
            "Managed": " (managed global instructions for all users)",
            "User": " (user's private global instructions for all projects)",
            "Project": " (project instructions, checked into the codebase)",
            "Local": " (user's private project instructions, not checked in)",
        }
        return labels.get(kind, "")

##############################
#### 5) 会话存储于记忆管理 #######
##############################
# 把完整 session 持久化到磁盘，支持恢复先前对话状态。
class SessionStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"

    def save(self, session: dict[str, Any]) -> Path:
        path = self.path(session["id"])
        path.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load(self, session_id: str) -> dict[str, Any]:
        return json.loads(self.path(session_id).read_text(encoding="utf-8"))
    # 找到最近一次保存的会话，支持 `--resume latest`。
    def latest(self) -> str | None:
        files = sorted(self.root.glob("*.json"), key=lambda item: item.stat().st_mtime)
        return files[-1].stem if files else None

class MiniAgent:
    def __init__(
        self,
        model_client,
        workspace: WorkspaceContext,
        session_store: SessionStore,
        session: dict[str, Any] | None = None,
        approval_policy: str = "ask",
        max_steps: int = 50,
        max_new_tokens: int = 12800,
        depth: int = 0,
        # 允许的最大委派深度，用来阻止子代理无限生成新的子代理。
        max_depth: int = 1,
        # 只读模式用于约束子代理，避免多个代理同时修改工作区。
        read_only: bool = False,
    ):
        self.model_client = model_client
        self.workspace = workspace
        self.session_store = session_store
        self.approval_policy = approval_policy
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.depth = depth
        self.max_depth = max_depth
        self.read_only = read_only
        self.scope_root = Path(workspace.repo_root).resolve()
        self.workdir = Path(workspace.cwd).resolve()
        self.aliases = {
            "list_files": "Glob",
            "read_file": "Read",
            "search": "Grep",
            "web_search": "WebSearch",
            "web_fetch": "WebFetch",
            "run_shell": "Bash",
            "write_file": "Write",
            "patch_file": "Edit",
            "delegate": "Agent",
        }
        self.session = session or {
            "id": datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
            "created_at": self.now(),
            "workspace_root": workspace.repo_root,
            "history": [],
            "memory": {"task": "", "files": [], "notes": [], "todos": []},
        }
        self.tools = self.build_tools()
        self.prefix = self.build_prefix()
        self.session_path = self.session_store.save(self.session)

    ############################################
    #### 2) 提示词的设计和cache #######
    ############################################
    # 构建“稳定提示前缀”：身份、环境、规则、工具、示例。
    # 这部分在同一个 MiniAgent 生命周期里通常不变，因此会缓存到 self.prefix。
    def build_prefix(self) -> str:
        tool_lines = []
        for name, tool in self.tools.items():
            fields = ", ".join(f"{key}: {value}" for key, value in tool["schema"].items())
            mode = "approval required" if tool["risky"] else "safe"
            tool_lines.append(f"- {name}({fields}) [{mode}] {tool['description']}")

        rules = "\n".join(
            [
                "- Use tools instead of guessing about the workspace.",
                "- Persist until the task is complete or you are truly blocked.",
                "- Prefer Glob, Grep, Read, WebFetch, WebSearch, Write, and Edit over Bash when possible.",
                f"- Use WebSearch for current or external information; include the current year ({date.today().year}) in latest-information queries when helpful.",
                "- Use WebFetch for a specific public URL. It may fail on authenticated or private pages.",
                "- Use TodoWrite for tasks with multiple meaningful steps.",
                "- Tool results and user messages may include <system-reminder> tags. They contain system context.",
                "- Return exactly one <tool>...</tool> or one <final>...</final> in each response.",
                '- JSON tool calls must look like: <tool>{"name":"ToolName","args":{...}}</tool>',
                "- For multi-line Write or Edit calls, prefer XML-style <tool ...> blocks.",
                "- Never invent tool results.",
                "- If you use WebSearch in the answer, include a Sources section with relevant markdown links.",
                "- Keep final answers concise and concrete.",
                "- Avoid repeating the exact same failed tool call unchanged.",
            ]
        )

        examples = "\n".join(
            [
                '<tool>{"name":"TodoWrite","args":{"todos":[{"content":"Inspect the target file","status":"in_progress","activeForm":"Inspecting the target file"}]}}</tool>',
                '<tool>{"name":"Glob","args":{"pattern":"*.py","path":"."}}</tool>',
                '<tool>{"name":"Read","args":{"file_path":"mini_claude_code.py","offset":1,"limit":80}}</tool>',
                '<tool>{"name":"Grep","args":{"pattern":"WorkspaceContext","path":"."}}</tool>',
                '<tool>{"name":"WebSearch","args":{"query":"latest bun documentation 2026","allowed_domains":["bun.sh"]}}</tool>',
                '<tool>{"name":"WebFetch","args":{"url":"https://example.com","prompt":"Summarize the page briefly."}}</tool>',
                '<tool>{"name":"Bash","args":{"command":"python -m py_compile mini_claude_code.py","timeout":20}}</tool>',
                '<tool name="Write" file_path="example.py"><content>def main():\n    return 0\n</content></tool>',
                '<tool name="Edit" file_path="example.py"><old_string>return 0</old_string><new_string>return 1</new_string></tool>',
                '<tool>{"name":"Agent","args":{"description":"inspect file","prompt":"Read mini_claude_code.py and summarize the main loop.","max_steps":3}}</tool>',
                "<final>Done.</final>",
            ]
        )

        sections = [
            "You are Claude Code, Anthropic's official CLI for Claude.",
            "You are helping with software engineering work inside a local workspace.",
            self.workspace.environment_prompt,
            "Rules:\n" + rules,
            "Tools:\n" + "\n".join(tool_lines),
            "Valid response examples:\n" + examples,
        ]
        return "\n\n".join(section for section in sections if section)

    def prompt(self, user_message: str) -> list[dict[str, str]]:
        system_prompt = "\n\n".join(self.workspace.append_system_context([self.prefix]))
        messages = [{"role": "system", "content": system_prompt}]
        meta = self.workspace.user_context_message()
        if meta:
            messages.append({"role": "user", "content": meta})
        messages.append(
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        self.memory_text(),
                        "Transcript:\n" + self.history_text(),
                        "Current user request:\n" + user_message,
                    ]
                ),
            }
        )
        return messages

    @staticmethod
    def parse(raw: str) -> tuple[str, Any]:
        raw = str(raw)
        has_json_tool = "<tool>" in raw and ("<final>" not in raw or raw.find("<tool>") < raw.find("<final>"))
        if has_json_tool:
            body = MiniAgent.extract(raw, "tool")
            try:
                payload = json.loads(body)
            except Exception:
                return "retry", MiniAgent.retry_notice("model returned malformed tool JSON")
            if not isinstance(payload, dict):
                return "retry", MiniAgent.retry_notice("tool payload must be a JSON object")
            if not str(payload.get("name", "")).strip():
                return "retry", MiniAgent.retry_notice("tool payload is missing a tool name")
            args = payload.get("args", {})
            if args is None:
                payload["args"] = {}
            elif not isinstance(args, dict):
                return "retry", MiniAgent.retry_notice("tool args must be a JSON object")
            return "tool", payload

        has_xml_tool = "<tool" in raw and ("<final>" not in raw or raw.find("<tool") < raw.find("<final>"))
        if has_xml_tool:
            payload = MiniAgent.parse_xml_tool(raw)
            return ("tool", payload) if payload is not None else ("retry", MiniAgent.retry_notice())

        if "<final>" in raw:
            final = MiniAgent.extract(raw, "final").strip()
            return ("final", final) if final else ("retry", MiniAgent.retry_notice("model returned an empty <final> answer"))

        raw = raw.strip()
        return ("final", raw) if raw else ("retry", MiniAgent.retry_notice("model returned an empty response"))

    @staticmethod
    def retry_notice(problem: str | None = None) -> str:
        prefix = f"Runtime notice: {problem}" if problem else "Runtime notice: model returned malformed tool output"
        return (
            f"{prefix}. Reply with a valid <tool> call or a non-empty <final> answer. "
            'For multi-line Write or Edit payloads, prefer <tool name="Write" file_path="file.py"><content>...</content></tool>.'
        )

    @staticmethod
    def parse_xml_tool(raw: str) -> dict[str, Any] | None:
        match = re.search(r"<tool(?P<attrs>[^>]*)>(?P<body>.*?)</tool>", raw, re.S)
        if not match:
            return None
        attrs = MiniAgent.parse_attrs(match.group("attrs"))
        name = str(attrs.pop("name", "")).strip()
        if not name:
            return None
        body = match.group("body")
        args = dict(attrs)
        for key in (
            "content",
            "old_string",
            "new_string",
            "command",
            "description",
            "prompt",
            "query",
            "url",
            "allowed_domains",
            "blocked_domains",
            "pattern",
            "path",
            "file_path",
        ):
            if f"<{key}>" in body:
                args[key] = MiniAgent.extract_raw(body, key)
        for key in ("allowed_domains", "blocked_domains"):
            if key in args and isinstance(args[key], str):
                args[key] = MiniAgent.normalize_domains(args[key])
        body_text = body.strip("\n")
        if name == "Write" and "content" not in args and body_text:
            args["content"] = body_text
        if name == "Agent" and "prompt" not in args and body_text:
            args["prompt"] = body_text.strip()
        return {"name": name, "args": args}

    @staticmethod
    def parse_attrs(text: str) -> dict[str, str]:
        attrs: dict[str, str] = {}
        for match in re.finditer(
            r"""([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:"([^"]*)"|'([^']*)')""",
            text,
        ):
            attrs[match.group(1)] = match.group(2) if match.group(2) is not None else match.group(3)
        return attrs

    @staticmethod
    def extract(text: str, tag: str) -> str:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        return text[start:] if end == -1 else text[start:end]

    @staticmethod
    def extract_raw(text: str, tag: str) -> str:
        return MiniAgent.extract(text, tag)


    ###############################################
    #### 3) 工具的访问与创建 ######
    ###############################################
    # 声明代理可用的工具白名单。
    # 每个工具都带有固定 schema、风险级别和执行入口，模型不能随意发明未注册工具。
    def build_tools(self) -> dict[str, dict[str, Any]]:
        tools = {
            "Glob": {
                "schema": {"pattern": "str", "path": "str='.'"},
                "description": "Find files by glob pattern under the workspace.",
                "risky": False,
                "run": self.tool_glob,
            },
            "Read": {
                "schema": {"file_path": "str", "offset": "int=1", "limit": "int=200"},
                "description": "Read a UTF-8 text file with line numbers.",
                "risky": False,
                "run": self.tool_read,
            },
            "Grep": {
                "schema": {"pattern": "str", "path": "str='.'", "glob": "str=''"},
                "description": "Search file contents with ripgrep or a fallback scan.",
                "risky": False,
                "run": self.tool_grep,
            },
            "WebFetch": {
                "schema": {"url": "str", "prompt": "str"},
                "description": "Fetch a public URL and extract content relevant to a prompt.",
                "risky": True,
                "run": self.tool_web_fetch,
            },
            "WebSearch": {
                "schema": {
                    "query": "str",
                    "allowed_domains": "list[str]=[]",
                    "blocked_domains": "list[str]=[]",
                },
                "description": "Search the public web for current information.",
                "risky": True,
                "run": self.tool_web_search,
            },
            "Bash": {
                "schema": {"command": "str", "timeout": "int=20", "description": "str=''"},
                "description": "Run a shell command from the primary working directory.",
                "risky": True,
                "run": self.tool_bash,
            },
            "Write": {
                "schema": {"file_path": "str", "content": "str"},
                "description": "Write or overwrite a UTF-8 text file.",
                "risky": True,
                "run": self.tool_write,
            },
            "Edit": {
                "schema": {
                    "file_path": "str",
                    "old_string": "str",
                    "new_string": "str",
                    "replace_all": "bool=false",
                },
                "description": "Replace exact text in a file.",
                "risky": True,
                "run": self.tool_edit,
            },
            "TodoWrite": {
                "schema": {"todos": "list[Todo]"},
                "description": "Update the session todo list for multi-step work.",
                "risky": False,
                "run": self.tool_todo_write,
            },
        }
        # 只有在没达到最大深度时才暴露 Agent 工具，避免递归委派失控。
        if self.depth < self.max_depth:
            tools["Agent"] = {
                "schema": {"description": "str", "prompt": "str", "max_steps": "int=3"},
                "description": "Ask a bounded read-only sub-agent to investigate.",
                "risky": False,
                "run": self.tool_agent,
            }
        return tools

    def resolve_tool_name(self, name: str) -> str:
        return self.aliases.get(name, name)

    def tool_example(self, name: str) -> str:
        examples = {
            "Glob": '<tool>{"name":"Glob","args":{"pattern":"*.py","path":"."}}</tool>',
            "Read": '<tool>{"name":"Read","args":{"file_path":"README.md","offset":1,"limit":80}}</tool>',
            "Grep": '<tool>{"name":"Grep","args":{"pattern":"MiniAgent","path":"."}}</tool>',
            "WebFetch": '<tool>{"name":"WebFetch","args":{"url":"https://example.com","prompt":"Summarize the page."}}</tool>',
            "WebSearch": '<tool>{"name":"WebSearch","args":{"query":"latest rust release 2026","allowed_domains":["blog.rust-lang.org"]}}</tool>',
            "Bash": '<tool>{"name":"Bash","args":{"command":"python -m py_compile mini_claude_code.py","timeout":20}}</tool>',
            "Write": '<tool name="Write" file_path="example.py"><content>def main():\n    return 0\n</content></tool>',
            "Edit": '<tool name="Edit" file_path="example.py"><old_string>return 0</old_string><new_string>return 1</new_string></tool>',
            "TodoWrite": '<tool>{"name":"TodoWrite","args":{"todos":[{"content":"Inspect the file","status":"in_progress","activeForm":"Inspecting the file"}]}}</tool>',
            "Agent": '<tool>{"name":"Agent","args":{"description":"inspect file","prompt":"Read the target file and summarize the relevant logic.","max_steps":3}}</tool>',
        }
        return examples.get(name, "")

    def validate_tool(self, name: str, args: dict[str, Any]) -> None:
        if name == "Glob":
            pattern = str(args.get("pattern", "")).strip()
            if not pattern:
                raise ValueError("pattern must not be empty")
            if not self.path(args.get("path", ".")).exists():
                raise ValueError("path does not exist")
            return

        if name == "Read":
            path = self.path(args.get("file_path", ""))
            if not path.is_file():
                raise ValueError("file_path is not a file")
            offset = self.as_int(args.get("offset"), 1)
            limit = self.as_int(args.get("limit"), 200)
            if offset < 1 or limit < 1:
                raise ValueError("offset and limit must be positive")
            return

        if name == "Grep":
            pattern = str(args.get("pattern", "")).strip()
            if not pattern:
                raise ValueError("pattern must not be empty")
            if not self.path(args.get("path", ".")).exists():
                raise ValueError("path does not exist")
            return

        if name == "WebFetch":
            url = str(args.get("url", "")).strip()
            prompt = str(args.get("prompt", "")).strip()
            if not url:
                raise ValueError("url must not be empty")
            if not prompt:
                raise ValueError("prompt must not be empty")
            parsed = urllib_parse.urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("url must be a valid http(s) URL")
            return

        if name == "WebSearch":
            query = str(args.get("query", "")).strip()
            if len(query) < 2:
                raise ValueError("query must be at least 2 characters")
            allowed_domains = self.normalize_domains(args.get("allowed_domains"))
            blocked_domains = self.normalize_domains(args.get("blocked_domains"))
            if allowed_domains and blocked_domains:
                raise ValueError("cannot specify both allowed_domains and blocked_domains")
            return

        if name == "Bash":
            command = str(args.get("command", "")).strip()
            timeout = self.as_int(args.get("timeout"), 20)
            if not command:
                raise ValueError("command must not be empty")
            if timeout < 1 or timeout > 120:
                raise ValueError("timeout must be in [1, 120]")
            return

        if name == "Write":
            path = self.path(args.get("file_path", ""))
            if path.exists() and path.is_dir():
                raise ValueError("file_path is a directory")
            if "content" not in args:
                raise ValueError("missing content")
            return

        if name == "Edit":
            path = self.path(args.get("file_path", ""))
            if not path.is_file():
                raise ValueError("file_path is not a file")
            old_string = str(args.get("old_string", ""))
            if not old_string:
                raise ValueError("old_string must not be empty")
            if "new_string" not in args:
                raise ValueError("missing new_string")
            text = path.read_text(encoding="utf-8", errors="replace")
            count = text.count(old_string)
            if count == 0:
                raise ValueError("old_string not found in file")
            if not self.as_bool(args.get("replace_all"), False) and count != 1:
                raise ValueError(f"old_string must occur exactly once, found {count}")
            return

        if name == "TodoWrite":
            todos = args.get("todos")
            if not isinstance(todos, list) or not todos:
                raise ValueError("todos must be a non-empty list")
            valid = {"pending", "in_progress", "completed"}
            for todo in todos:
                if not isinstance(todo, dict):
                    raise ValueError("each todo must be an object")
                if not str(todo.get("content", "")).strip():
                    raise ValueError("todo content must not be empty")
                if todo.get("status") not in valid:
                    raise ValueError("todo status must be pending, in_progress, or completed")
                if not str(todo.get("activeForm", "")).strip():
                    raise ValueError("todo activeForm must not be empty")
            return

        if name == "Agent":
            # 子代理只能在受限深度内创建，避免委派树无限扩张。
            if self.depth >= self.max_depth:
                raise ValueError("delegate depth exceeded")
            if not str(args.get("description", "")).strip():
                raise ValueError("description must not be empty")
            if not str(args.get("prompt", "")).strip():
                raise ValueError("prompt must not be empty")
            max_steps = self.as_int(args.get("max_steps"), 3)
            if max_steps < 1 or max_steps > 8:
                raise ValueError("max_steps must be in [1, 8]")
            return

        raise ValueError(f"unsupported tool {name}")

    def repeated_tool_call(self, name: str, args: dict[str, Any]) -> bool:
        events = [item for item in self.session["history"] if item["role"] == "tool"]
        if len(events) < 2:
            return False
        recent = events[-2:]
        return all(item["name"] == name and item["args"] == args for item in recent)

    def approve(self, name: str, args: dict[str, Any]) -> bool:
        if self.read_only:
            return False
        if self.approval_policy == "auto":
            return True
        if self.approval_policy == "never":
            return False
        try:
            answer = input(f"approve {name} {json.dumps(args, ensure_ascii=True)}? [y/N] ")
        except EOFError:
            return False
        return answer.strip().lower() in {"y", "yes"}

    def run_tool(self, name: str, args: dict[str, Any]) -> str:
        canonical = self.resolve_tool_name(name)
        tool = self.tools.get(canonical)
        if tool is None:
            return f"error: unknown tool '{name}'"
        args = args or {}
        try:
            self.validate_tool(canonical, args)
        except Exception as exc:
            message = f"error: invalid arguments for {canonical}: {exc}"
            example = self.tool_example(canonical)
            return message if not example else f"{message}\nexample: {example}"
        if self.repeated_tool_call(canonical, args):
            return (
                f"error: repeated identical tool call for {canonical}; "
                "choose a different tool or return a final answer"
            )
        if tool["risky"] and not self.approve(canonical, args):
            return f"error: approval denied for {canonical}"
        try:
            # 所有工具输出在返回模型前统一裁剪，防止日志/命令结果无限增长。
            return clip(tool["run"](args))
        except Exception as exc:
            return f"error: tool {canonical} failed: {exc}"

    def path(self, raw_path: Any) -> Path:
        raw = str(raw_path).strip()
        if not raw:
            raise ValueError("path must not be empty")
        path = Path(raw)
        resolved = (path if path.is_absolute() else self.workdir / path).resolve()
        try:
            resolved.relative_to(self.scope_root)
        except Exception as exc:
            raise ValueError(f"path escapes workspace: {raw}") from exc
        return resolved

    def display_path(self, path: Path) -> str:
        for base in (self.workdir, self.scope_root):
            try:
                return str(path.relative_to(base))
            except Exception:
                pass
        return str(path)

    def should_ignore_path(self, path: Path) -> bool:
        try:
            parts = path.relative_to(self.scope_root).parts
        except Exception:
            parts = path.parts
        return any(part in IGNORED_PATH_NAMES for part in parts)

    def tool_glob(self, args: dict[str, Any]) -> str:
        pattern = str(args.get("pattern", "")).strip()
        base = self.path(args.get("path", "."))
        if base.is_file():
            matches = [base] if base.match(pattern) else []
        else:
            matches = [item for item in base.glob(pattern) if not self.should_ignore_path(item)]
        return "\n".join(self.display_path(path) for path in matches[:200]) or "(no matches)"

    def tool_read(self, args: dict[str, Any]) -> str:
        path = self.path(args["file_path"])
        offset = self.as_int(args.get("offset"), 1)
        limit = self.as_int(args.get("limit"), 200)
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        excerpt = lines[offset - 1 : offset - 1 + limit]
        body = "\n".join(f"{number:>6}\t{line}" for number, line in enumerate(excerpt, start=offset))
        return f"# {self.display_path(path)}\n{body}"

    def tool_grep(self, args: dict[str, Any]) -> str:
        pattern = str(args.get("pattern", "")).strip()
        base = self.path(args.get("path", "."))
        glob_filter = str(args.get("glob", "")).strip()
        if shutil.which("rg"):
            command = ["rg", "-n", "--smart-case", "--max-count", "200"]
            if glob_filter:
                command.extend(["--glob", glob_filter])
            command.extend([pattern, str(base)])
            result = subprocess.run(command, cwd=self.workdir, capture_output=True, text=True, check=False)
            return result.stdout.strip() or result.stderr.strip() or "(no matches)"

        matches: list[str] = []
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            regex = re.compile(re.escape(pattern), re.IGNORECASE)

        files = [base] if base.is_file() else [item for item in base.rglob("*") if item.is_file()]
        for file_path in files:
            if self.should_ignore_path(file_path):
                continue
            if glob_filter and not file_path.match(glob_filter):
                continue
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            for number, line in enumerate(lines, start=1):
                if regex.search(line):
                    matches.append(f"{self.display_path(file_path)}:{number}:{line}")
                    if len(matches) >= 200:
                        return "\n".join(matches)
        return "\n".join(matches) or "(no matches)"

    def tool_web_fetch(self, args: dict[str, Any]) -> str:
        start = time.time()
        url = str(args["url"]).strip()
        prompt = str(args["prompt"]).strip()
        final_url, status, reason, body, content_type = self.make_request(url)
        raw_text = self.decode_body(body, content_type)
        title = self.html_title(raw_text)
        result_text = self.html_to_text(raw_text) if "html" in content_type.lower() else clip(raw_text, MAX_WEB_TEXT)
        payload = {
            "url": url,
            "finalUrl": final_url,
            "title": title,
            "bytes": len(body),
            "code": status,
            "codeText": reason,
            "durationMs": int((time.time() - start) * 1000),
            "contentType": content_type,
            "result": f"Prompt: {prompt}\n\nFetched content:\n{result_text}",
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def tool_web_search(self, args: dict[str, Any]) -> str:
        start = time.time()
        query = str(args["query"]).strip()
        allowed_domains = self.normalize_domains(args.get("allowed_domains"))
        blocked_domains = self.normalize_domains(args.get("blocked_domains"))
        query_variants = [query]
        if allowed_domains:
            query_variants.append(query + " " + " ".join(f"site:{domain}" for domain in allowed_domains))

        last_response = {
            "query": query,
            "searchUrl": "",
            "engine": "duckduckgo-lite",
            "status": 0,
            "statusText": "",
            "results": [],
        }
        for variant in query_variants:
            for base_url, engine in (
                ("https://lite.duckduckgo.com/lite/?", "duckduckgo-lite"),
                ("https://html.duckduckgo.com/html/?", "duckduckgo-html"),
            ):
                search_url = base_url + urllib_parse.urlencode({"q": variant})
                final_url, status, reason, body, content_type = self.make_request(search_url)
                raw_html = self.decode_body(body, content_type)
                results = self.parse_search_results(raw_html, allowed_domains, blocked_domains)
                last_response = {
                    "query": query,
                    "searchUrl": final_url,
                    "engine": engine,
                    "status": status,
                    "statusText": reason,
                    "results": results,
                }
                if results:
                    payload = dict(last_response)
                    payload["durationSeconds"] = round(time.time() - start, 3)
                    return json.dumps(payload, ensure_ascii=False, indent=2)

        payload = {
            **last_response,
            "durationSeconds": round(time.time() - start, 3),
        }
        payload["note"] = "No results matched the query and domain filters."
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def tool_bash(self, args: dict[str, Any]) -> str:
        command = str(args.get("command", "")).strip()
        timeout = self.as_int(args.get("timeout"), 20)
        result = subprocess.run(
            command,
            cwd=self.workdir,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return "\n".join(
            [
                f"exit_code: {result.returncode}",
                "stdout:",
                result.stdout.strip() or "(empty)",
                "stderr:",
                result.stderr.strip() or "(empty)",
            ]
        )

    def tool_write(self, args: dict[str, Any]) -> str:
        path = self.path(args["file_path"])
        content = str(args["content"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"wrote {self.display_path(path)} ({len(content)} chars)"

    def tool_edit(self, args: dict[str, Any]) -> str:
        path = self.path(args["file_path"])
        old_string = str(args["old_string"])
        new_string = str(args["new_string"])
        replace_all = self.as_bool(args.get("replace_all"), False)
        text = path.read_text(encoding="utf-8", errors="replace")
        count = text.count(old_string)
        if count == 0:
            raise ValueError("old_string not found in file")
        if replace_all:
            path.write_text(text.replace(old_string, new_string), encoding="utf-8")
            return f"patched {self.display_path(path)} ({count} replacements)"
        if count != 1:
            raise ValueError(f"old_string must occur exactly once, found {count}")
        path.write_text(text.replace(old_string, new_string, 1), encoding="utf-8")
        return f"patched {self.display_path(path)}"

    def tool_todo_write(self, args: dict[str, Any]) -> str:
        old_todos = list(self.session["memory"].get("todos", []))
        todos = args["todos"]
        all_done = all(todo["status"] == "completed" for todo in todos)
        # todos 属于 working memory 的结构化状态，而不是 transcript 的替代品。
        self.session["memory"]["todos"] = [] if all_done else todos
        if todos and not self.session["memory"]["task"]:
            self.session["memory"]["task"] = clip(str(todos[0]["content"]), 300)
        return json.dumps(
            {"oldTodos": old_todos, "newTodos": self.session["memory"]["todos"]},
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def as_int(value: Any, default: int) -> int:
        if value is None or value == "":
            return default
        return int(value)

    @staticmethod
    def as_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def normalize_domains(values: Any) -> list[str]:
        if not values:
            return []
        if isinstance(values, str):
            values = [part.strip() for part in values.split(",")]
        return [str(value).strip().lower() for value in values if str(value).strip()]

    @staticmethod
    def host_matches_domain(hostname: str, domain: str) -> bool:
        hostname = hostname.lower()
        domain = domain.lower()
        return hostname == domain or hostname.endswith("." + domain)

    @staticmethod
    def make_request(url: str, timeout: int = DEFAULT_WEB_TIMEOUT) -> tuple[str, int, str, bytes, str]:
        req = urllib_request.Request(
            url,
            headers={
                "User-Agent": NETWORK_USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        try:
            with urllib_request.urlopen(req, timeout=timeout) as response:
                # 网络响应体也要限长，避免大网页或二进制内容直接灌爆上下文。
                body = response.read(MAX_WEB_BYTES + 1)
                if len(body) > MAX_WEB_BYTES:
                    body = body[:MAX_WEB_BYTES]
                status = getattr(response, "status", response.getcode())
                reason = getattr(response, "reason", "") or ""
                content_type = response.headers.get("Content-Type", "")
                return response.geturl(), int(status), str(reason), body, content_type
        except urllib_error.HTTPError as exc:
            body = exc.read(MAX_WEB_BYTES + 1)[:MAX_WEB_BYTES]
            return exc.geturl(), int(exc.code), str(exc.reason or ""), body, exc.headers.get("Content-Type", "")

    @staticmethod
    def decode_body(body: bytes, content_type: str) -> str:
        charset_match = re.search(r"charset=([A-Za-z0-9._-]+)", content_type, re.I)
        encoding = charset_match.group(1) if charset_match else "utf-8"
        try:
            return body.decode(encoding, errors="replace")
        except LookupError:
            return body.decode("utf-8", errors="replace")

    @staticmethod
    def html_to_text(raw_html: str) -> str:
        text = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw_html)
        text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
        text = re.sub(r"(?is)<!--.*?-->", " ", text)
        text = re.sub(r"(?i)<br\s*/?>", "\n", text)
        text = re.sub(r"(?i)</(p|div|li|section|article|h[1-6]|tr)>", "\n", text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = html.unescape(text)
        text = re.sub(r"\r", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        # HTML 转正文后仍然再裁一次，防止抓到超长页面正文。
        return clip(text.strip(), MAX_WEB_TEXT)

    @staticmethod
    def html_title(raw_html: str) -> str:
        match = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw_html)
        if not match:
            return ""
        return re.sub(r"\s+", " ", html.unescape(match.group(1))).strip()

    @staticmethod
    def resolve_search_href(href: str) -> str:
        href = href.strip()
        if href.startswith("//"):
            href = "https:" + href
        elif href.startswith("/"):
            href = "https://duckduckgo.com" + href
        parsed = urllib_parse.urlparse(href)
        if parsed.netloc.endswith("duckduckgo.com"):
            uddg = urllib_parse.parse_qs(parsed.query).get("uddg")
            if uddg:
                return urllib_parse.unquote(uddg[0])
        return href

    @staticmethod
    def parse_search_results(raw_html: str, allowed_domains: list[str], blocked_domains: list[str]) -> list[dict[str, str]]:
        results: list[dict[str, str]] = []
        seen: set[str] = set()
        patterns = [
            re.compile(
                r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                re.I | re.S,
            ),
            re.compile(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.I | re.S),
        ]
        for pattern in patterns:
            for href, title_html in pattern.findall(raw_html):
                url = MiniAgent.resolve_search_href(href)
                parsed = urllib_parse.urlparse(url)
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    continue
                hostname = parsed.hostname or ""
                if blocked_domains and any(MiniAgent.host_matches_domain(hostname, domain) for domain in blocked_domains):
                    continue
                if allowed_domains and not any(MiniAgent.host_matches_domain(hostname, domain) for domain in allowed_domains):
                    continue
                title = re.sub(r"(?s)<[^>]+>", " ", title_html)
                title = re.sub(r"\s+", " ", html.unescape(title)).strip()
                if not title or url in seen:
                    continue
                seen.add(url)
                results.append({"title": title, "url": url})
                if len(results) >= 8:
                    return results
            if results:
                return results
        return results

    #####################################################
    #### 4) 上下文压缩 #####
    #####################################################
    def history_text(self) -> str:
        history = self.session["history"]
        if not history:
            return "- empty"
        lines: list[str] = []
        # 对较旧的 Read 结果做去重，避免同一文件因为早期多次读取而反复占用上下文。
        seen_reads: set[str] = set()
        # 只保留最近几条事件更高保真；更老的事件会被更激进地压缩。
        recent_start = max(0, len(history) - 6)
        for index, item in enumerate(history):
            recent = index >= recent_start
            if item["role"] == "tool" and item["name"] in {"Write", "Edit"}:
                path = str(item["args"].get("file_path", ""))
                if path:
                    seen_reads.discard(path)
            if item["role"] == "tool" and item["name"] == "Read" and not recent:
                path = str(item["args"].get("file_path", ""))
                if path in seen_reads:
                    continue
                seen_reads.add(path)
            if item["role"] == "tool":
                # 近期工具结果保留更多字符，旧结果只保留较短摘要。
                lines.append(
                    f"[tool:{item['name']}] {json.dumps(item['args'], ensure_ascii=False, sort_keys=True)}"
                )
                lines.append(clip(item["content"], 900 if recent else 180))
            else:
                # assistant/user 历史同样按“近大远小”的策略压缩。
                lines.append(f"[{item['role']}] {clip(item['content'], 900 if recent else 220)}")
        return clip("\n".join(lines), MAX_HISTORY)

    ##############################
    #### 5) 会话存储于记忆管理 #######
    ##############################
    @classmethod
    def from_session(
        cls,
        model_client,
        workspace: WorkspaceContext,
        session_store: SessionStore,
        session_id: str,
        **kwargs,
    ) -> "MiniAgent":
        return cls(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            session=session_store.load(session_id),
            **kwargs,
        )

    @staticmethod
    def remember(bucket: list[str], item: str, limit: int) -> None:
        if not item:
            return
        if item in bucket:
            bucket.remove(item)
        bucket.append(item)
        del bucket[:-limit]

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def memory_text(self) -> str:
        # 这里读取的是提炼后的 working memory，而不是完整 transcript。
        memory = self.session["memory"]
        todos = memory.get("todos", [])
        todo_text = (
            "\n".join(
                f"- [{item['status']}] {item['content']} ({item['activeForm']})" for item in todos
            )
            if todos
            else "- none"
        )
        notes = "\n".join(f"- {item}" for item in memory["notes"]) or "- none"
        files = ", ".join(memory["files"]) or "-"
        return "\n".join(
            [
                "Working memory:",
                f"- task: {memory['task'] or '-'}",
                f"- files: {files}",
                "- todos:",
                todo_text,
                "- notes:",
                notes,
            ]
        )

    def record(self, item: dict[str, Any]) -> None:
        self.session["history"].append(item)
        self.session_path = self.session_store.save(self.session)

    def note_tool(self, name: str, args: dict[str, Any], result: str) -> None:
        memory = self.session["memory"]
        path = args.get("file_path")
        if name in {"Read", "Write", "Edit"} and path:
            self.remember(memory["files"], str(path), 8)
        # 工作记忆里的 note 只保留短摘要，避免工具长输出再次进入上下文。
        note = f"{name}: {clip(result.replace(chr(10), ' '), 220)}"
        self.remember(memory["notes"], note, 5)

    def ask(self, user_message: str) -> str:
        memory = self.session["memory"]
        # task 是 working memory 里的显式小状态，用来跨轮维持任务连续性。
        if not memory["task"]:
            memory["task"] = clip(user_message, 300)
        # 用户消息先进入完整 transcript，作为永久事件存档。
        self.record({"role": "user", "content": user_message, "created_at": self.now()})

        attempts = 0
        tool_steps = 0
        max_attempts = max(self.max_steps * 3, self.max_steps + 4)

        while tool_steps < self.max_steps and attempts < max_attempts:
            attempts += 1
            raw = self.model_client.complete(self.prompt(user_message), self.max_new_tokens)
            kind, payload = self.parse(raw)

            if kind == "tool":
                tool_steps += 1
                name = payload.get("name", "")
                args = payload.get("args", {})
                result = self.run_tool(name, args)
                canonical = self.resolve_tool_name(name)
                self.record(
                    {
                        "role": "tool",
                        "name": canonical,
                        "args": args,
                        "content": result,
                        "created_at": self.now(),
                    }
                )
                self.note_tool(canonical, args, result)
                continue

            if kind == "retry":
                self.record({"role": "assistant", "content": payload, "created_at": self.now()})
                continue

            final = (payload or raw).strip()
            self.record({"role": "assistant", "content": final, "created_at": self.now()})
            self.remember(memory["notes"], clip(final, 220), 5)
            return final

        final = (
            "Stopped after too many malformed model responses without a valid tool call or final answer."
            if attempts >= max_attempts and tool_steps < self.max_steps
            else "Stopped after reaching the step limit without a final answer."
        )
        self.record({"role": "assistant", "content": final, "created_at": self.now()})
        return final

    def reset(self) -> None:
        self.session["history"] = []
        self.session["memory"] = {"task": "", "files": [], "notes": [], "todos": []}
        self.session_path = self.session_store.save(self.session)

    ###################################################
    #### 6) 使用（有界）子代理进行委派** ##########
    ###################################################
    # 创建一个有界子代理，把局部调查任务委派出去。
    # 当前实现是同步执行，但通过 depth / max_depth / read_only 施加了明确边界。
    # 子代理工具：在边界受限的前提下把局部调查委托给只读子 agent。
    def tool_agent(self, args: dict[str, Any]) -> str:
        # 第一层边界：最大委派深度。
        if self.depth >= self.max_depth:
            raise ValueError("delegate depth exceeded")
        # 第二层边界：子代理继承上下文，但被强制设成只读，且禁止审批高风险操作。
        child = MiniAgent(
            model_client=self.model_client,
            workspace=self.workspace,
            session_store=self.session_store,
            approval_policy="never",
            max_steps=self.as_int(args.get("max_steps"), 10),
            max_new_tokens=self.max_new_tokens,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            read_only=True,
        )
        # 子代理只拿到一个提炼后的任务描述和压缩历史，而不是无约束地重新探索全局状态。
        child.session["memory"]["task"] = clip(str(args["description"]), 300)
        child.session["memory"]["notes"] = [clip(self.history_text(), 400)]
        return "agent_result:\n" + child.ask(str(args["prompt"]))

def build_agent(args: argparse.Namespace) -> MiniAgent:
    workspace = WorkspaceContext.build(args.cwd)
    store = SessionStore(Path(workspace.repo_root) / ".mini-coding-agent" / "sessions")
    api_key = (args.api_key or os.getenv("DASHSCOPE_API_KEY") or "").strip() or None
    model_client = DashScopeModelClient(
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.request_timeout,
        enable_thinking=not args.disable_thinking,
    )
    # 连接 session 恢复链路：支持显式 session id，也支持 `latest`。
    session_id = args.resume
    if session_id == "latest":
        session_id = store.latest()
    if session_id:
        return MiniAgent.from_session(
            model_client=model_client,
            workspace=workspace,
            session_store=store,
            session_id=session_id,
            approval_policy=args.approval,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
        )
    return MiniAgent(
        model_client=model_client,
        workspace=workspace,
        session_store=store,
        approval_policy=args.approval,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
    )

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Minimal Claude Code-style coding agent for DashScope-compatible chat models.",
    )
    parser.add_argument("prompt", nargs="*", help="Optional one-shot prompt.")
    parser.add_argument("--cwd", default=".", help="Primary working directory.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name.")
    parser.add_argument("--api-key", default=None, help="API key; falls back to DASHSCOPE_API_KEY.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="DashScope-compatible OpenAI base URL.")
    parser.add_argument("--request-timeout", type=int, default=300, help="Model request timeout in seconds.")
    parser.add_argument("--resume", default=None, help="Session id to resume or 'latest'.")
    parser.add_argument(
        "--approval",
        choices=("ask", "auto", "never"),
        default="ask",
        help="Approval policy for risky tools.",
    )
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum tool/model iterations per request.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum model output tokens per step.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling value.")
    parser.add_argument("--disable-thinking", action="store_true", help="Disable reasoning mode if supported.")
    return parser

def main(argv=None) -> int:
    init_env()
    args = build_arg_parser().parse_args(argv)
    try:
        agent = build_agent(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(build_welcome(agent, model=args.model, endpoint=args.base_url))

    if args.prompt:
        prompt = " ".join(args.prompt).strip()
        if prompt:
            print()
            try:
                print(agent.ask(prompt))
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
                return 1
        return 0

    while True:
        try:
            user_input = input("\nmini-claude-code> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            return 0
        if user_input == "/help":
            print(HELP_DETAILS)
            continue
        if user_input == "/memory":
            print(agent.memory_text())
            continue
        if user_input == "/session":
            print(agent.session_path)
            continue
        if user_input == "/reset":
            agent.reset()
            print("session reset")
            continue

        print()
        try:
            print(agent.ask(user_input))
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)

if __name__ == "__main__":
    raise SystemExit(main())
