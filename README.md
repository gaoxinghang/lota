# lota

用 LangChain 复刻隔壁 `learn-claude-code/s01_agent_loop` 的核心功能：
一个 Shell 工具，一个显式循环。

模型决定是否调用工具，程序执行命令并回传结果；没有工具调用就结束本轮。
不使用预构建 Agent，保留可直接阅读的循环实现。

## 文件结构

- `lota/model.py`：读取配置，通过 `init_chat_model` 创建模型。
- `lota/tools.py`：用 `@tool` 定义 Shell 工具。
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

可以输入“列出当前目录中的 Python 文件”，观察命令及输出，再继续追问。
输入 `q`、`exit`，或在输入时按 Ctrl-C / Ctrl-D 退出。

## 核心流程

```text
用户输入 → 模型回复 → 有工具调用 → 执行命令 → 追加结果 → 再问模型
                  └→ 无工具调用 → 打印回复，等待下一轮输入
```

四个关键接口：

- `init_chat_model(...)`：统一创建模型，业务代码不导入厂商专用类。
- `bind_tools(TOOLS)`：把工具定义提供给模型，不会自动执行工具。
- `invoke(messages)`：传入完整历史，返回含文本或工具调用的消息。
- `ToolMessage(tool_call_id=...)`：通过调用 ID 配对工具结果。

系统提示词放在第一条 `SystemMessage` 中；用户输入用 `HumanMessage`；
模型回复直接追加到历史。一次返回多个工具调用时，按顺序执行。

模型提供方固定为 `openai`，无需配置 `MODEL_PROVIDER`。
切换 OpenAI 兼容服务只需修改连接配置和模型名称，不改 Agent 循环。

## 执行边界

- Shell 命令在启动程序时的工作目录执行，每次启动独立子进程；
  单次命令中的 `cd` 不会改变后续调用的目录。
- 命令超时为 120 秒；回传输出最多 50000 字符，终端预览前 200 字符。
- 没有审批、沙箱或可靠的命令安全检查，也没有循环次数限制。
  超时与截断不是安全边界。
- 相比原章，未保留可绕过的命令子串黑名单及 readline 输入补丁。

**程序会执行模型生成的 Shell 命令。请在隔离的测试环境中运行，
不要授予敏感文件或凭据的访问权限。**

模型接口说明见 [LangChain Models 文档][models]。

[models]: https://docs.langchain.com/oss/python/langchain/models
