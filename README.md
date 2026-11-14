# lota

用 LangChain 复刻隔壁 `learn-claude-code/s02_tool_use` 的核心功能：
五个工具，一个显式循环。

模型决定是否调用工具，程序按名称分发并回传结果；没有工具调用就结束本轮。
不使用预构建 Agent，保留可直接阅读的循环实现。

## 文件结构

- `lota/model.py`：读取配置，通过 `init_chat_model` 创建模型。
- `lota/tools.py`：用 `@tool` 定义 Shell 与文件工具，统一注册到 `TOOLS`。
- `lota/loop.py`：调用模型、执行工具、追加工具结果。
- `lota/main.py`：命令行交互，跨轮保留消息历史。

建议按以上顺序阅读。

## 快速开始

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp .env.example .env
```

项目仅使用 OpenAI 协议，支持 OpenAI 官方及兼容服务。
`requirements.txt` 已包含 `langchain-openai`，无需另行安装集成包。

然后填写 `.env`：

| 配置项 | 含义 |
| --- | --- |
| `MODEL_ID` | 该服务中支持工具调用的模型名称，必填 |
| `API_KEY` | 必填，作为 `api_key` 传给集成包 |
| `BASE_URL` | 必填，作为 `base_url` 传给集成包 |

请在项目根目录运行，程序读取当前目录的 `.env`，不展开变量引用。
配置直接传给 LangChain，不做额外校验；缺失值的处理由集成包决定，
可能使用其默认配置或环境变量，也可能报错。
使用兼容服务时，填写其 `BASE_URL`、密钥及模型名称，
并确保服务支持 OpenAI 格式的工具调用。

```sh
lota
```

安装会在当前 Python 环境中注册 `lota` 命令，使用前需激活该环境。
可编辑安装后，修改源码无需重新安装。原来的 `python -m lota.main`
仍然可用。

可以输入“查找所有 Python 文件，再读取 README.md”，观察工具调用及输出。
输入 `q`、`exit`，或在输入时按 Ctrl-C / Ctrl-D 退出。

## 核心流程

```text
用户输入 → 模型回复 → 有工具调用 → 按名称分发 → 追加结果 → 再问模型
                  └→ 无工具调用 → 打印回复，等待下一轮输入
```

四个关键接口：

- `init_chat_model(...)`：统一创建模型，业务代码不导入厂商专用类。
- `bind_tools(TOOLS)`：把工具定义提供给模型，不会自动执行工具。
- `invoke(messages)`：传入完整历史，返回含文本或工具调用的消息。
- `ToolMessage(tool_call_id=...)`：通过调用 ID 配对工具结果。

系统提示词放在第一条 `SystemMessage` 中；用户输入用 `HumanMessage`；
模型回复直接追加到历史。一次返回多个工具调用时，按顺序执行。
未知工具、参数校验和文件操作错误以 `Error:` 结果回传，让模型调整调用。

模型提供方固定为 `openai`，无需配置 `MODEL_PROVIDER`。
切换 OpenAI 兼容服务只需修改连接配置和模型名称，不改 Agent 循环。

## 工具扩展（s02）

| 工具 | 用途 |
| --- | --- |
| `bash` | 执行 Shell 命令 |
| `read_file` | 读取 UTF-8 文本，可用 `limit` 限制返回行数 |
| `write_file` | 写入或覆盖 UTF-8 文本，自动创建父目录 |
| `edit_file` | 精确替换第一处匹配文本，不存在则报错 |
| `glob` | 按模式查找路径，支持 `**` 递归，最多返回 200 条 |

`@tool` 从函数签名与文档字符串生成工具定义，不手写 JSON Schema。
`TOOL_MAP` 从 `TOOLS` 自动生成，循环只按工具名查表并调用 `invoke`。
新增工具只需定义函数并加入 `TOOLS`，无需修改循环。

文件工具通过 `safe_path` 检查路径，拒绝越界路径及指向外部的符号链接。
`glob` 也过滤指向工作目录外的匹配项。

## 执行边界

- Shell 命令在启动程序时的工作目录执行，每次启动独立子进程；
  单次命令中的 `cd` 不会改变后续调用的目录。
- Shell 命令超时为 120 秒，回传输出最多 50000 字符。
  文件读取默认不截断；所有工具的终端预览只显示前 200 字符。
- 没有审批、沙箱或可靠的命令安全检查，也没有循环次数限制。
  超时与截断不是安全边界。
- 文件路径检查不是沙箱：不能防止检查后路径被替换，Bash 也不受其限制。
- 相比原章，未保留可绕过的命令子串黑名单及 readline 输入补丁。

**程序会执行模型生成的 Shell 命令。请在隔离的测试环境中运行，
不要授予敏感文件或凭据的访问权限。**

## 测试

使用标准库 `unittest`，不调用模型 API；文件操作只在临时目录中进行。

```sh
python -m unittest discover -s tests -v
```

模型接口说明见 [LangChain Models 文档][models]。

[models]: https://docs.langchain.com/oss/python/langchain/models
