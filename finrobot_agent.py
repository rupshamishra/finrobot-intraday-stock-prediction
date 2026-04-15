import autogen
from finrobot.utils import get_current_date, register_keys_from_json
from finrobot.agents.workflow import SingleAssistant

# Load OpenAI config
llm_config = {
    "config_list": autogen.config_list_from_json("OAI_CONFIG_LIST"),
    "temperature": 0,
}

# Register finance API keys
register_keys_from_json("config_api_keys")

agent = SingleAssistant(
    "Market_Analyst",
    llm_config,
    human_input_mode="NEVER",
)

company = "AAPL"

agent.chat(
    f"Use all tools to analyze {company} as of {get_current_date()}. "
    "Summarize positives, risks, and give a short outlook."
)
