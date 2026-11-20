# lota

用 LangChain 复刻隔壁 `learn-claude-code` 新版 s01–s03 的基础能力，
并加入 s05 的 TODO 清单、s06 的子 Agent 和 s07 的按需技能加载，
不另造 s04 的 Hook 注册表。s08 目前只实现第一步：大工具结果转存。

模型决定是否调用工具，`create_agent` 负责分发工具并回传结果；
没有工具调用就结束本轮。工具日志和权限检查通过 `wrap_tool_call` 中间件接入。

## 文件结构

- `lota/model.py`：读取配置，通过 `init_chat_model` 创建模型。
- `lota/tools/`：Shell（`shell.py`）、文件（`files.py`）、清单（`todo.py`）、
  子 Agent（`subagent.py`）和技能（`skills.py`）工具。
- `lota/middleware/`：工具日志、权限审批和工具结果预算。
- `lota/workspace.py`：工具与权限检查共用的工作目录。
- `lota/agent.py`：通过 `create_agent` 注册工具和中间件，配置串行执行。
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
输入 `q`、`exit`，或在主输入提示处按 Ctrl-C / Ctrl-D 退出。
权限审批提示处只有 `y` / `yes`（不区分大小写）允许执行；
其他输入或 Ctrl-D 均拒绝本次调用；审批处 Ctrl-C 不保证只拒绝本次调用。

## 核心流程

```text
用户输入 → 模型回复 → 无工具调用 → 打印回复，等待下一轮输入
                  └→ 有工具调用 → 参数校验 → 权限检查
                                             ├→ 允许 → 执行工具
                                             └→ 拒绝 → 不执行
                              追加工具结果 ←────┘
                                   └→ 再问模型
```

关键接口：

- `init_chat_model(...)`：创建模型，保留现有连接配置。
- `create_agent(...)`：绑定工具，管理模型调用、工具分发及结果配对。
- `wrap_tool_call`：日志中间件只为父 Agent 打印工具名、参数和结果预览；
  权限中间件校验参数、审批并回传错误。
- `agent.invoke({"messages": history})`：运行一轮并返回完整消息历史。

中文系统提示词由 `create_agent(system_prompt=...)` 配置，不存入对话历史。
用户输入用 `HumanMessage`；每轮结束后用 Agent 返回的 `messages` 更新历史。
`max_concurrency=1` 保持同批工具按请求顺序执行，不并发审批或操作文件。
未知工具、参数校验和文件操作错误以 `Error:` 结果回传，让模型调整调用。

命令行和中间件均使用同步接口，通过 `invoke` 运行 Agent，
通过 `handler(request)` 执行工具，不创建事件循环。
框架仍可能在线程池中执行工具，串行执行不代表在主线程执行。

模型提供方固定为 `openai`，无需配置 `MODEL_PROVIDER`。
切换 OpenAI 兼容服务只需修改连接配置和模型名称，不改 Agent 配置。

## 工具扩展（s02）

| 工具 | 用途 |
| --- | --- |
| `bash` | 执行 Shell 命令 |
| `read_file` | 读取 UTF-8 文本，可用 `limit` 限制返回行数 |
| `write_file` | 写入或覆盖 UTF-8 文本，自动创建父目录 |
| `edit_file` | 精确替换第一处匹配文本，不存在则报错 |
| `glob` | 按模式查找路径，支持 `**` 递归，最多返回 200 条 |

`@tool` 从函数签名与文档字符串生成工具定义，不手写 JSON Schema。
`lota/agent.py` 在 `build_agent()` 中注册父 Agent 的工具，
其中 `todo_write` 绑定当前会话的独立清单。新增工具时，在 `lota/tools/` 定义后
加入注册列表；有审批需求时，再到 `lota/middleware/permission.py` 添加规则。

## 权限管线（s03）

父 Agent 的日志中间件先打印工具名和完整参数。父子 Agent 的工具参数都要通过
LangChain 工具 Schema 校验，再进入三道闸门：

| 闸门 | 匹配条件 | 行为 |
| --- | --- | --- |
| 硬拒绝 | 命令包含拒绝列表中的子串 | 直接拒绝，不询问用户 |
| 规则匹配 | 文件越界或潜在破坏性命令 | 转交用户审批 |
| 用户审批 | 显示审批原因并询问用户 | 仅明确批准后执行 |

硬拒绝列表沿用原章：`rm -rf /`、`sudo`、`shutdown`、`reboot`、
`mkfs`、`dd if=`、`> /dev/sda`。硬拒绝优先，用户无法通过审批放行。

需要审批的情况：

- `read_file`、`write_file`、`edit_file` 的目标解析后位于工作目录外，
  包括相对越界、绝对越界和指向外部的符号链接。
- `glob` 的模式路径解析后位于工作目录外，例如 `../outside/*.py`。
- Shell 命令匹配 `rm` / `del` 命令位置规则（不区分大小写），
  或包含原章的 `rm `、`> /etc/`、`chmod 777` 子串。

没有命中规则则直接执行。每次调用单独审批，不记忆上次批准。
拒绝时回传 `Permission denied.`，仍使用原调用 ID 配对，
不影响同批后续工具调用。实际审批由程序在执行工具前完成。

与 s02 不同，工作区外的读写、编辑和 Glob 搜索现在可以经批准执行。
路径权限统一由 `check_rules` 检查，工具自身不再限制路径或过滤外部结果；
直接调用工具的 `invoke` 会绕过中间件中的审批。

可在隔离环境中试用：创建工作区文件直接通过，删除文件触发审批，
读取工作区外文件触发审批，调用 `sudo` 则直接拒绝。

## TODO 清单（s05）

系统提示词建议多步骤任务使用 `todo_write`，具体何时调用由模型决定。
工具每次替换整张清单，空列表可清空。每项要有非空描述和状态：
`pending`（待办）、`in_progress`（进行中）、`completed`（已完成）。
最多 20 项、同时最多一项进行中。更新失败不覆盖旧清单。
清单完整返回给模型，终端只显示统一的结果预览。
清单保留在当前 Agent 实例中，退出不保存。

## 子 Agent（s06）

`run_subagent(prompt)` 同步委派子任务。每次调用从一条新消息开始，
不接收父对话历史；父 Agent 只收到最终文本，不收到子工具调用和结果。
子 Agent 可使用父 Agent 除 `run_subagent` 外的全部工具，共享工作区和审批规则。
父子各自维护 TODO 清单，子清单保留在同一个子 Agent 实例中。
子 Agent 不打印工具日志；工具结果仍回传给子模型，权限提示照常显示。
子 Agent 最多执行 30 个 LangGraph 图步，达到上限时向父 Agent 返回错误提示。

## 技能加载（s07）

在工作区的 `skills/<技能目录>/SKILL.md` 放置 UTF-8 Markdown，
开头用 YAML 头部声明非空的 `name` 和 `description`：

```markdown
---
name: code-review
description: 审查代码改动并找出回归风险。
---

# 代码审查

先确认改动目标，再检查行为变化和测试覆盖。
```

启动时读取并缓存技能文件，但父子 Agent 的系统提示只包含名称和简介。
模型调用 `load_skill(name)` 后，完整的 `SKILL.md` 才作为工具结果进入
当前对话。子 Agent 不继承父对话，需要时自行加载。没有可用技能时不注册该工具；
无效文件和指向技能目录外的文件会跳过，同名技能按路径排序只保留第一份并发出警告。
修改技能后需要重启 CLI。

可以在本仓库启动 `lota`，输入「请先加载 code-review 技能，再审查当前改动」。
加载只返回文本，不执行脚本。简介和加载后的正文会发送给模型，
请只使用可信的技能文件，不要在技能里放凭据等敏感信息。

## 工具结果预算（s08 第一步）

父子 Agent 每次调用模型前，只检查最新一批工具结果。文本总量超过
200,000 字符时，从最大的结果开始，把单条超过 30,000 字符的结果转存到
工作区的 `.task_outputs/tool-results/`，消息中只保留相对路径和前 2,000 字符。
降到预算内，或没有符合条件的结果时停止。模型可用 `read_file` 取回原文；
工具调用与结果的配对不变，后续历史保留的是预览。

这一步不会额外调用模型，也不裁剪或总结旧历史。转存文件不会自动清理，
可能包含敏感内容，不应提交到 Git。

## 执行边界

- Shell 命令在启动程序时的工作目录执行，每次启动独立子进程；
  单次命令中的 `cd` 不会改变后续调用的目录。
- Shell 命令超时为 120 秒，回传输出最多 50000 字符。
  文件读取工具默认不截断；终端日志只显示前 200 字符，不参与工具结果预算。
- 权限规则是教学示例，不是完整 Shell 解析器，会有误判和漏判；
  命令子串黑名单可以绕过，不构成可靠的命令安全检查。
- 没有沙箱；父 Agent 的框架步数上限设为 `sys.maxsize`，
  避免默认上限提前结束任务。超时与截断不是安全边界。
- Glob 只检查模式路径，不逐个检查展开后的匹配项；通配符可能匹配到
  外部符号链接，递归搜索也可能沿链接访问工作区外，且不会因此额外审批。
- 文件路径检查不能防止检查后路径被替换，Bash 也不受路径规则限制。
- 启动时加载原章的 readline 输入补丁；未安装 readline 时跳过。
  配置面向 GNU Readline，libedit 后端不保证兼容。

**程序会执行模型生成的 Shell 命令。请在隔离的测试环境中运行，
不要授予敏感文件或凭据的访问权限。**

## 测试

使用标准库 `unittest`，不调用模型 API；文件操作只在临时目录中进行。
Shell 执行和用户输入均使用模拟，覆盖硬拒绝、单次审批、越界文件访问、
参数校验和工具结果配对，不执行真实危险命令。
Agent 集成测试还覆盖同批工具顺序、跨轮历史、拒绝审批后继续执行、
清单校验和跨轮保留，以及技能目录和按需加载。

```sh
python -m unittest discover -s tests -v
```

接口说明见 [Models][models]、[Agents][agents] 和 [Middleware][middleware]。

[models]: https://docs.langchain.com/oss/python/langchain/models
[agents]: https://docs.langchain.com/oss/python/langchain/agents
[middleware]: https://docs.langchain.com/oss/python/langchain/middleware/custom
