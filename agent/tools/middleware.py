"""
Compatibility notes for the LangChain 0.1.x implementation.

The original project used the newer LangChain middleware API:
wrap_tool_call, before_model and dynamic_prompt. The dependency set in this
repository currently pins LangChain 0.1.20, where that middleware package does
not exist. Prompt switching is therefore handled in agent.react_agent by
selecting a normal or report AgentExecutor before execution starts.
"""

from utils.logger_handler import logger
from utils.prompt_loader import load_report_prompts, load_system_prompts


def select_prompt(is_report: bool = False) -> str:
    if is_report:
        logger.info("[prompt] report prompt selected")
        return load_report_prompts()

    logger.info("[prompt] normal prompt selected")
    return load_system_prompts()
