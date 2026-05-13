import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_handler import prompts_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


def _load_prompt(config_key: str, label: str) -> str:
    try:
        prompt_path = get_abs_path(prompts_conf[config_key])
    except KeyError as e:
        logger.error(f"[{label}]在yaml配置项中没有{config_key}配置项")
        raise e

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"[{label}]解析提示词出错，{str(e)}")
        raise e


def load_system_prompts() -> str:
    return _load_prompt("main_prompt_path", "load_system_prompts")


def load_rag_prompts() -> str:
    return _load_prompt("rag_summarize_prompt_path", "load_rag_prompts")


def load_report_prompts() -> str:
    return _load_prompt("report_prompt_path", "load_report_prompts")


if __name__ == "__main__":
    print(load_report_prompts())
