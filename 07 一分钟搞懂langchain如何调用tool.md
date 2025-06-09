一个典型langchain的使用流程：
```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools, initialize_agent, AgentType
  
# 初始化语言模型
llm = OpenAI(temperature=0.9)

# 创建提示模板
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

# 创建链
chain = LLMChain(llm=llm, prompt=prompt)


# 我们在此处加载工具 
# "llm-math" 是一个简单的计算器工具 
tools = load_tools(["llm-math"], llm=llm) 
# 初始化Agent 
# 我们使用 AgentType.ZERO_SHOT_REACT_DESCRIPTION，它会根据工具的描述来决定使用哪个工具 
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
```

![alt text](image/07.png)


## 一、🧰初始化流程
工具的初始化通常直接创建具体的工具类实例，或者使用 `load_tools` 函数加载预定义的工具：
1. 每个工具都继承自 `BaseTool` 类
2. 工具的定义包括名称、描述和执行函数
代码路径：`libs/core/langchain_core/tools/base.py`

```python

class BaseTool(Runnable[str, str], BaseModel, ABC):

    """工具的基类"""
    name: str
    description: str
    return_direct: bool = False
    verbose: bool = False
    # ...

```


##  二、调用流程

当代理使用工具时：
1. 代理选择一个工具并提供输入
2. `AgentExecutor` 调用 `_perform_agent_action` 方法
3. 该方法获取工具实例并调用其 `__call__` 方法
4. `__call__` 方法会调用工具的 `run` 方法，后者会调用 `_run` 方法
5. `_run` 方法包含工具的具体实现逻辑，例如执行搜索、计算等操作


`__call__` 方法
```python
@deprecated("0.1.47", alternative="invoke", removal="1.0")

    def __call__(self, tool_input: str, callbacks: Callbacks = None) -> str:
        """Make tool callable."""
        return self.run(tool_input, callbacks=callbacks)
```
关键逻辑是调用工具类的`run`方法


工具`run`方法
```python
def run(
        self,
        tool_input: Union[str, dict[str, Any]],
        verbose: Optional[bool] = None,
        start_color: Optional[str] = "green",
        color: Optional[str] = "green",
        callbacks: Callbacks = None,
        *,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        run_name: Optional[str] = None,
        run_id: Optional[uuid.UUID] = None,
        config: Optional[RunnableConfig] = None,
        tool_call_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:

        callback_manager = CallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose or bool(verbose),
            tags,
            self.tags,
            metadata,
            self.metadata,
        )

        run_manager = callback_manager.on_tool_start(
            {"name": self.name, "description": self.description},
            tool_input if isinstance(tool_input, str) else str(tool_input),
            color=start_color,
            name=run_name,
            run_id=run_id,
            inputs=tool_input if isinstance(tool_input, dict) else None,
            **kwargs,
        )

  

        content = None
        artifact = None
        status = "success"
        error_to_raise: Union[Exception, KeyboardInterrupt, None] = None
        try:
            child_config = patch_config(config, callbacks=run_manager.get_child())
            with set_config_context(child_config) as context:
                tool_args, tool_kwargs = self._to_args_and_kwargs(
                    tool_input, tool_call_id
                )

                ...... # 校验处理tool参数

                response = context.run(self._run, *tool_args, **tool_kwargs)

            if self.response_format == "content_and_artifact":
                if not isinstance(response, tuple) or len(response) != 2:
                    msg = (
                        "Since response_format='content_and_artifact' "
                        "a two-tuple of the message content and raw tool output is 
                        f"expected. Instead generated response of type: "
                        f"{type(response)}."
                    )
                    error_to_raise = ValueError(msg)
                else:
                    content, artifact = response
            else:
                content = response
        except (ValidationError, ValidationErrorV1) as e:

            ...... #处理异常情况 

        output = _format_output(content, artifact, tool_call_id, self.name, status)
        # 记录工具执行结束
        run_manager.on_tool_end(output, color=color, name=self.name, **kwargs)    
        return output
```
1. **配置回调函数：**
    - 通过`callback_manager = CallbackManager.configure(...)`设置运行工具后的回调
2. **启动工具执行管理：**
    - 通过 `run_manager = callback_manager.on_tool_start(...)`设置工具执行时的记录
3. **核心逻辑位于`try...except`块内部：**
    - `child_config = patch_config(...)`并`with set_config_context(...)`将前面初始化的内容配置到工具执行上下文中。
    - `tool_args, tool_kwargs = self._to_args_and_kwargs(...)`将输入给tool的`tool_input`参数转化为底层函数（即`self._run`）所需的参数和关键字参数。
    - `response = context.run(self._run, *tool_args, **tool_kwargs)`**真正的工作**发生在这里。这一行调用了实际的内部函数 ( `self._run`)，该函数执行该工具设计的任何功能（例如，从 API 获取天气数据）。
4. **处理不同类型的工具响应：**
    - `if self.response_format == "content_and_artifact":`检查工具是否返回以下内容：内容消息 ( `content`) 和一些详细数据 ( `artifact`)。例如，`content`内容是“北京天气晴朗”， `artifact`是来自天气 API 的原始 JSON 响应。若格式不符合预期，返回错误。
    - 判断格式不为`content_and_artifact`，它只内容消息`content`。
5. **格式化并返回结果：**
    - 通过`output = _format_output(content, artifact, ...)`将tool执行结果转化成标准格式。
    - 最后，`return output`将工具执行的结果发送回调用它的人。

output = _format_output(content, artifact, tool_call_id, self.name, status)
代码路径：`libs/langchain/langchain/agents/agent.py`




context.run是如何运行的
```python
@contextmanager

def set_config_context(config: RunnableConfig) -> Generator[Context, None, None]:

    """Set the child Runnable config + tracing context.
    Args:
        config (RunnableConfig): The config to set.
    """

    from langsmith.run_helpers import _set_tracing_context
    ctx = copy_context()
    config_token, _ = ctx.run(_set_config_context, config)

    try:
        yield ctx
    finally:

        ctx.run(var_child_runnable_config.reset, config_token)
        ctx.run(
            _set_tracing_context,

            {
                "parent": None,
                "project_name": None,
                "tags": None,
                "metadata": None,
                "enabled": None,
                "client": None,
            },
        )
```

context.run来源于set_config_context函数返回的Generator对象。
类似于**上下文管理器**。可以把它当成代码的临时“工作区”。在临时工作区以某种配置执行，李艾历史工作区后又恢复为原来的配置。
- **`@contextmanager`**：这是 Python 中一个特殊的装饰器。意味着我们可以将此函数与`with`语句一起使用，例如`with set_config_context(my_config):`。它可以保证在进入`with`代码块时执行某些设置操作，并在离开代码块时执行一些清理操作，即使发生错误也是如此。
- **保存当前“规则”（`copy_context()`和`_set_config_context`）**：
    - `ctx = copy_context()`：在使用新配置之前，代码会先对_当前_执行环境的上下文进行快照。用于便于后续还原。
    - `config_token, _ = ctx.run(_set_config_context, config)`：将`config`（函数接收到临时配置参数）应用到系统。它还会返回一个`config_token`，它类似于一个特殊的“撤消”功能，提供一个途径让系统恢复到之前的状态。
- **使用临时规则运行代码（`yield ctx`）**：
    - `yield ctx`：这是上下文管理器的核心。当我们使用 时`with set_config_context(my_config):`，块内的代码`with`会紧接着执行`yield`。在此期间，你的代码将按照传入`config`的配置运行。

## 三、总结

 简单来说，Agent 通过LLM进行**思考**并**选择**工具，而工具则负责**执行**具体的任务，并将最终结果反回给 Agent，以便Agent进行下一步行动。
