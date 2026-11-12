use std::{
    collections::{HashMap, HashSet},
    fmt,
    sync::{Arc, OnceLock, Weak},
};

use async_openai::types::chat::ChatCompletionTools;

use crate::{
    agent::{
        callback::{
            after_tool_callback::AfterToolCallback, before_llm_callback::BeforeLlmCallback,
            before_tool_callback::BeforeToolCallback,
        },
        context::ExecutionContext,
        event::{AgentEvent, ContentItem, ToolResult, ToolResultStatus},
        tool_confirmation::ToolConfirmation,
    },
    constant::{AGENT_NAME, FINAL_ANSWER_TOOL_NAME},
    llm::llm_request::{LlmRequest, ToolChoice},
    session::session_manager::{InMemorySessionManager, SessionManager},
    tool::{Tool, ToolBox, transfer::TransferTool},
};

pub struct Agent {
    pub name: String,
    /// 这个 agent 能做什么的简短描述。
    ///
    /// 当 agent 被 [`crate::tool::agent_tool::AgentTool`] 包装后，这段文本会
    /// 成为协调员看到的工具描述，因此应该说明**什么时候**该把任务委派给它。
    pub description: String,
    pub model: String,
    pub instructions: Vec<String>,
    pub toolbox: Arc<ToolBox>,
    pub max_steps: usize,
    pub before_tool_callbacks: Vec<Arc<dyn BeforeToolCallback>>,
    pub after_tool_callbacks: Vec<Arc<dyn AfterToolCallback>>,
    pub before_llm_callbacks: Vec<Arc<dyn BeforeLlmCallback>>,
    pub session_manager: Box<dyn SessionManager>,

    /// 禁止转移到兄弟节点。父节点和子节点不受影响。
    pub disallow_transfer_to_peers: bool,

    /// 子节点。使用 `OnceLock` 是因为 Rust 里必须先有 `Arc<Self>`，
    /// 才能建立“子节点指向父节点”的弱引用，因此树只能在构造完成后组装。
    sub_agents: OnceLock<Vec<Arc<Agent>>>,

    /// 父节点的弱引用。用 `Weak` 避免 `Arc` 父子循环导致内存泄漏。
    parent: OnceLock<Weak<Agent>>,
}

impl Agent {
    pub fn new(model: String, instructions: Vec<String>, toolbox: Arc<ToolBox>) -> Self {
        Self {
            model,
            instructions,
            toolbox,
            max_steps: 10,
            name: AGENT_NAME.to_owned(),
            description: String::new(),
            before_tool_callbacks: Vec::new(),
            after_tool_callbacks: Vec::new(),
            before_llm_callbacks: Vec::new(),
            session_manager: Box::new(InMemorySessionManager::new()),
            disallow_transfer_to_peers: false,
            sub_agents: OnceLock::new(),
            parent: OnceLock::new(),
        }
    }

    pub fn with_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    /// 当这个 agent 被当作工具使用时，展示给协调员的描述。
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// 禁止转移到兄弟节点（父节点和子节点仍然可以转移）。
    pub fn with_disallow_transfer_to_peers(mut self, disallow: bool) -> Self {
        self.disallow_transfer_to_peers = disallow;
        self
    }

    /// 组装 agent 树：把 `sub_agents` 挂到当前 agent 下面。
    ///
    /// Python 可以在构造函数里直接给子节点设置 `parent`，但 Rust 必须先有
    /// `Arc<Self>` 才能创建 `Weak<Self>`。因此树的组装被拆成两步：
    ///
    /// ```ignore
    /// let math = Arc::new(Agent::new(...).with_name("math".into()));
    /// let router = Arc::new(Agent::new(...).with_name("router".into()));
    /// router.set_sub_agents(vec![math])?;
    /// ```
    ///
    /// 校验规则与书中一致：同一父节点下名称不能重复，一个 agent 只能有一个
    /// 父节点。这里额外检查整棵树内的名称唯一性以及环形引用，因为这两点
    /// 会直接影响按名称转移和 `Arc` 的生命周期。
    pub fn set_sub_agents(self: &Arc<Self>, sub_agents: Vec<Arc<Agent>>) -> anyhow::Result<()> {
        if self.sub_agents.get().is_some() {
            anyhow::bail!("agent '{}' already has sub-agents", self.name);
        }

        let mut direct_names = HashSet::new();
        for sub in &sub_agents {
            if !direct_names.insert(sub.name.clone()) {
                anyhow::bail!("duplicate sub-agent name: '{}'", sub.name);
            }
            if Arc::ptr_eq(sub, self) {
                anyhow::bail!("agent '{}' cannot be its own sub-agent", self.name);
            }
        }

        // 一个 agent 不能有两个父节点。先整体检查，再统一写入，
        // 避免校验失败时留下“挂了一半”的树。
        for sub in &sub_agents {
            if sub.parent.get().is_some() {
                anyhow::bail!(
                    "agent '{}' already has a parent; an agent can only appear once in the tree",
                    sub.name
                );
            }
        }

        // 名称在整个树中必须唯一，否则 transfer_to_agent("name") 会有歧义。
        let mut root = self.clone();
        while let Some(parent) = root.parent() {
            root = parent;
        }
        let mut names = HashMap::new();
        root.collect_names(&mut names)?;
        for sub in &sub_agents {
            if sub.contains_agent(Arc::as_ptr(self)) {
                anyhow::bail!(
                    "cannot attach '{}' under '{}': it would create a cycle",
                    sub.name,
                    self.name
                );
            }
            sub.collect_names(&mut names)?;
        }

        self.sub_agents
            .set(sub_agents.clone())
            .map_err(|_| anyhow::anyhow!("agent '{}' already has sub-agents", self.name))?;

        let parent = Arc::downgrade(self);
        for sub in sub_agents {
            // 前面已经确认所有子节点都没有 parent，这里不会失败。
            let _ = sub.parent.set(parent.clone());
        }

        Ok(())
    }

    /// 当前节点的子节点列表；未组装时返回空切片。
    pub fn sub_agents(&self) -> &[Arc<Agent>] {
        self.sub_agents.get().map(Vec::as_slice).unwrap_or_default()
    }

    /// 当前节点的父节点。弱引用升级失败说明父节点已被释放。
    pub fn parent(&self) -> Option<Arc<Agent>> {
        self.parent.get().and_then(Weak::upgrade)
    }

    /// 当前 agent 允许转移到的全部目标。
    ///
    /// 三类目标与书中一致：
    /// 1. 子节点；
    /// 2. 父节点；
    /// 3. 兄弟节点（可通过 `disallow_transfer_to_peers` 关闭）。
    pub fn transfer_targets(&self) -> Vec<Arc<Agent>> {
        let mut targets = self.sub_agents().to_vec();

        if let Some(parent) = self.parent() {
            targets.push(parent.clone());

            if !self.disallow_transfer_to_peers {
                for sibling in parent.sub_agents() {
                    if !std::ptr::eq(sibling.as_ref(), self) {
                        targets.push(sibling.clone());
                    }
                }
            }
        }

        // 理论上树结构已经保证不会重复；这里仍按名称去重，
        // 保证生成的 enum 中不会出现重复值。
        let mut seen = HashSet::new();
        targets.retain(|agent| seen.insert(agent.name.clone()));
        targets
    }

    /// 在**允许的转移目标**中按名字查找。
    ///
    /// 书中从根节点搜索整棵树。这里直接搜索转移目标列表，效果相同且更安全：
    /// 不在目标列表里的 agent 本来就不允许通过工具转移过去。
    pub fn find_transfer_target(&self, name: &str) -> Option<Arc<Agent>> {
        self.transfer_targets()
            .into_iter()
            .find(|agent| agent.name == name)
    }

    fn collect_names(&self, names: &mut HashMap<String, usize>) -> anyhow::Result<()> {
        let ptr = self as *const Agent as usize;
        if let Some(existing) = names.get(&self.name) {
            if *existing == ptr {
                anyhow::bail!(
                    "agent '{}' appears more than once in the agent tree",
                    self.name
                );
            }
            anyhow::bail!("duplicate agent name '{}' in agent tree", self.name);
        }
        names.insert(self.name.clone(), ptr);

        for sub in self.sub_agents() {
            sub.collect_names(names)?;
        }
        Ok(())
    }

    fn contains_agent(&self, target: *const Agent) -> bool {
        if std::ptr::eq(self, target) {
            return true;
        }
        self.sub_agents()
            .iter()
            .any(|sub| sub.contains_agent(target))
    }

    pub fn with_before_tool_callbacks(mut self, callback: Arc<dyn BeforeToolCallback>) -> Self {
        self.before_tool_callbacks.push(callback);
        self
    }

    pub fn with_after_tool_callbacks(mut self, callback: Arc<dyn AfterToolCallback>) -> Self {
        self.after_tool_callbacks.push(callback);
        self
    }

    pub fn with_before_llm_callbacks(mut self, callback: Arc<dyn BeforeLlmCallback>) -> Self {
        self.before_llm_callbacks.push(callback);
        self
    }

    pub fn with_session_manager(mut self, session_manager: impl SessionManager + 'static) -> Self {
        self.session_manager = Box::new(session_manager);
        self
    }

    pub async fn prepare_llm_request(
        &self,
        context: &mut ExecutionContext,
    ) -> anyhow::Result<LlmRequest> {
        if !context.session.pending_tool_calls.is_empty()
            && context.session.state.contains_key("tool_confirmations")
        {
            let confirmation_results = self.process_confirmations(context).await?;

            if !confirmation_results.is_empty() {
                context.session.add_event(AgentEvent::new(
                    context.execution_id.clone(),
                    "tool",
                    confirmation_results,
                ));
            }

            context.session.pending_tool_calls.clear();
            context.session.state.remove("tool_confirmations");
        }

        let tool_choice = if self.has_final_answer_tool() {
            ToolChoice::Required
        } else {
            ToolChoice::Auto
        };

        let content_items = context
            .session
            .events
            .iter()
            .flat_map(|ev| ev.content.iter().cloned())
            .collect();
        let mut tools = self.tool_definitions();
        if let Some(transfer_tool) = self.transfer_tool_definition() {
            tools.push(transfer_tool);
        }
        let mut llm_request = LlmRequest::new(self.instructions.clone(), content_items)
            .with_tools(tools)
            .with_tool_choice(tool_choice);

        for before_llm_callback in &self.before_llm_callbacks {
            before_llm_callback
                .before_llm_call(&mut context.session, &mut llm_request)
                .await
        }

        Ok(llm_request)
    }

    async fn process_confirmations(
        &self,
        context: &ExecutionContext,
    ) -> anyhow::Result<Vec<ContentItem>> {
        let pending_calls = &context.session.pending_tool_calls;
        let confirmations: Vec<ToolConfirmation> = serde_json::from_value(
            context
                .session
                .state
                .get("tool_confirmations")
                .unwrap()
                .clone(),
        )?;

        let confirmation_map: HashMap<String, ToolConfirmation> = confirmations
            .into_iter()
            .map(|c| (c.tool_call_id.clone(), c))
            .collect();

        let mut results = Vec::with_capacity(pending_calls.len());

        for pending_call in pending_calls {
            let tool_call_id = pending_call.tool_call.id.clone();
            let name = pending_call.tool_call.function.name.clone();
            let confirmation = confirmation_map.get(&tool_call_id);

            let (status, content) = match confirmation {
                Some(c) if c.approved => {
                    let mut arguments: serde_json::Map<String, serde_json::Value> =
                        serde_json::from_str(&pending_call.tool_call.function.arguments)?;
                    if let Some(overrides) = &c.modified_arguments {
                        arguments.extend(overrides.clone());
                    }
                    let args_json = serde_json::Value::Object(arguments).to_string();

                    match self.toolbox.get(&name) {
                        Some(tool) => match tool.execute(&args_json, context).await {
                            Ok(result) => {
                                tracing::info!(
                                    "Confirmed tool call ({tool_call_id}): {name} ok: {result}"
                                );
                                (ToolResultStatus::Success, result)
                            }
                            Err(err) => {
                                tracing::warn!(
                                    "Confirmed tool call ({tool_call_id}): {name} failed: {err}"
                                );
                                (ToolResultStatus::Error, format!("Tool execution error: {err}"))
                            }
                        },
                        None => (
                            ToolResultStatus::Error,
                            format!("Tool execution error: unknown tool {name}"),
                        ),
                    }
                }
                Some(c) => {
                    let reason = c
                        .reason
                        .clone()
                        .unwrap_or_else(|| "Tool execution was rejected by user.".to_string());
                    (ToolResultStatus::Error, reason)
                }
                None => (
                    ToolResultStatus::Error,
                    "Tool execution was not approved.".to_string(),
                ),
            };

            results.push(ContentItem::ToolResult(ToolResult::new(
                tool_call_id,
                name,
                status,
                content,
            )));
        }

        Ok(results)
    }

    pub fn tool_definitions(&self) -> Vec<ChatCompletionTools> {
        self.toolbox
            .iter()
            .filter_map(|t| t.1.definition().ok())
            .collect()
    }

    /// 如果当前 agent 有可转移目标，生成 `transfer_to_agent` 的工具定义。
    ///
    /// 工具定义每步都重新生成，保证 agent 树组装完成后再运行时，
    /// 父节点/兄弟节点一定会被列进 enum。
    pub fn transfer_tool_definition(&self) -> Option<ChatCompletionTools> {
        let targets = self.transfer_targets();
        if targets.is_empty() {
            return None;
        }

        match TransferTool::new(targets).definition() {
            Ok(definition) => Some(definition),
            Err(err) => {
                tracing::warn!("failed to build transfer tool definition: {err}");
                None
            }
        }
    }

    pub fn has_final_answer_tool(&self) -> bool {
        self.toolbox.contains_key(FINAL_ANSWER_TOOL_NAME)
    }
}

#[derive(Debug)]
pub struct AgentResult<T> {
    pub output: Option<T>,
    pub context: ExecutionContext,
    pub status: AgentStatus,
}

impl<T> AgentResult<T> {
    pub fn new(output: Option<T>, context: ExecutionContext, status: AgentStatus) -> Self {
        Self {
            output,
            context,
            status,
        }
    }
}

impl<T: fmt::Display> fmt::Display for AgentResult<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.output {
            Some(output) => write!(f, "{output}"),
            None => write!(f, "No output"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentStatus {
    Complete,
    PendingConfirmation,
    Error,
}

#[cfg(test)]
mod transfer_tree_tests {
    use super::*;

    fn agent(name: &str) -> Arc<Agent> {
        Arc::new(
            Agent::new(
                "test-model".to_owned(),
                vec![format!("You are {name}.")],
                Arc::new(ToolBox::new()),
            )
            .with_name(name.to_owned())
            .with_description(format!("{name} description")),
        )
    }

    fn names(agents: &[Arc<Agent>]) -> Vec<String> {
        agents.iter().map(|agent| agent.name.clone()).collect()
    }

    #[test]
    fn parent_and_siblings_are_transfer_targets() {
        let math = agent("math");
        let search = agent("search");
        let router = agent("router");
        router
            .set_sub_agents(vec![math.clone(), search.clone()])
            .unwrap();

        assert_eq!(names(&router.transfer_targets()), ["math", "search"]);
        assert_eq!(
            names(&math.transfer_targets()),
            ["router", "search"],
            "math should reach its parent and its sibling"
        );
        assert!(math.parent().is_some());
        assert!(math.find_transfer_target("nobody").is_none());
    }

    #[tokio::test]
    async fn llm_request_includes_the_dynamic_transfer_tool() {
        let math = agent("math");
        let search = agent("search");
        let router = agent("router");
        router
            .set_sub_agents(vec![math.clone(), search.clone()])
            .unwrap();

        let mut context = ExecutionContext::default();
        let request = router.prepare_llm_request(&mut context).await.unwrap();
        let tools = serde_json::to_string(&request.tools).unwrap();
        assert!(tools.contains("transfer_to_agent"));
        assert!(tools.contains("\"math\""));
        assert!(tools.contains("\"search\""));

        // 叶子节点同样会得到父节点/兄弟节点的转移工具。
        let request = math.prepare_llm_request(&mut context).await.unwrap();
        let tools = serde_json::to_string(&request.tools).unwrap();
        assert!(tools.contains("\"router\""));
        assert!(tools.contains("\"search\""));
    }

    #[test]
    fn disallow_transfer_to_peers_removes_siblings() {
        let math = Arc::new(
            Agent::new(
                "test-model".to_owned(),
                vec!["You are math.".to_owned()],
                Arc::new(ToolBox::new()),
            )
            .with_name("math".to_owned())
            .with_disallow_transfer_to_peers(true),
        );
        let search = agent("search");
        let router = agent("router");
        router.set_sub_agents(vec![math.clone(), search]).unwrap();

        assert_eq!(names(&math.transfer_targets()), ["router"]);
    }

    #[test]
    fn an_agent_cannot_have_two_parents() {
        let math = agent("math");
        let router = agent("router");
        let other = agent("other");
        router.set_sub_agents(vec![math.clone()]).unwrap();

        assert!(other.set_sub_agents(vec![math]).is_err());
    }

    #[test]
    fn duplicate_names_across_branches_are_rejected() {
        let left = agent("left");
        let right = agent("right");
        let root = agent("root");
        root.set_sub_agents(vec![left.clone(), right.clone()])
            .unwrap();

        left.set_sub_agents(vec![agent("duplicate")]).unwrap();
        assert!(right.set_sub_agents(vec![agent("duplicate")]).is_err());
    }

    #[test]
    fn cycles_are_rejected() {
        let child = agent("child");
        let root = agent("root");
        root.set_sub_agents(vec![child.clone()]).unwrap();

        assert!(child.set_sub_agents(vec![root]).is_err());
    }
}
