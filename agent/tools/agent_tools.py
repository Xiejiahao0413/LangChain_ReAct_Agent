import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.tools import tool

from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


_rag_service = None
external_data = {}

USER_IDS = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010"]
MONTHS = [
    "2025-01",
    "2025-02",
    "2025-03",
    "2025-04",
    "2025-05",
    "2025-06",
    "2025-07",
    "2025-08",
    "2025-09",
    "2025-10",
    "2025-11",
    "2025-12",
]


def _get_rag_service() -> RagSummarizeService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RagSummarizeService()
    return _rag_service


@tool
def rag_summarize(query: str) -> str:
    """Search the local vector store for robot vacuum reference material."""
    return _get_rag_service().rag_summarize(query)


@tool
def get_weather(city: str) -> str:
    """Get demo weather information for a city."""
    return f"城市{city}天气为晴天，气温26摄氏度，空气湿度50%，南风1级，AQI21，最近6小时降雨概率极低"


@tool
def get_user_location(ignored_input: str = "") -> str:
    """Get the current user's demo city. The input is ignored."""
    return random.choice(["深圳", "合肥", "杭州"])


@tool
def get_user_id(ignored_input: str = "") -> str:
    """Get the current demo user's ID. The input is ignored."""
    return random.choice(USER_IDS)


@tool
def get_current_month(ignored_input: str = "") -> str:
    """Get a demo month that exists in the sample usage data. The input is ignored."""
    return random.choice(MONTHS)


def generate_external_data():
    if external_data:
        return

    external_data_path = get_abs_path(agent_conf["external_data_path"])
    if not Path(external_data_path).exists():
        raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

    with open(external_data_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)

        for row in reader:
            if len(row) < 6:
                logger.warning(f"[fetch_external_data]跳过无效外部数据行：{row}")
                continue

            user_id, feature, efficiency, consumables, comparison, month = [cell.strip() for cell in row[:6]]
            if not user_id or not month:
                logger.warning(f"[fetch_external_data]跳过缺少用户或月份的数据行：{row}")
                continue

            external_data.setdefault(user_id, {})[month] = {
                "特征": feature,
                "效率": efficiency,
                "耗材": consumables,
                "对比": comparison,
            }


def _parse_user_month(action_input: str) -> tuple[str, str]:
    action_input = action_input.strip()
    if action_input.startswith("{"):
        payload = json.loads(action_input)
        return str(payload["user_id"]).strip(), str(payload["month"]).strip()

    if "," in action_input:
        user_id, month = action_input.split(",", 1)
        return user_id.strip(), month.strip()

    raise ValueError('fetch_external_data input must be JSON or "user_id,month"')


@tool
def fetch_external_data(action_input: str) -> str:
    """Fetch usage data. Input must be JSON {"user_id":"1001","month":"2025-01"} or "1001,2025-01"."""
    generate_external_data()
    user_id, month = _parse_user_month(action_input)

    try:
        return json.dumps(external_data[user_id][month], ensure_ascii=False)
    except KeyError:
        logger.warning(f"[fetch_external_data]未检索到用户{user_id}在{month}的使用记录")
        return ""


@tool
def fill_context_for_report(ignored_input: str = "") -> str:
    """Mark the current task as a report-generation task. The input is ignored."""
    return "报告生成上下文已准备"
