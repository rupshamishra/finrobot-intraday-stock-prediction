import autogen

llm_config = {
    "config_list": autogen.config_list_from_json("OAI_CONFIG_LIST"),
    "temperature": 0.7,
}

assistant = autogen.AssistantAgent(
    name="SimpleAgent",
    llm_config=llm_config,
)

assistant.initiate_chat(
    recipient=assistant,
    message="Explain what an AI agent is in very simple words."
)
