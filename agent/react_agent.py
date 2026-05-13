import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from agent.tools.agent_tools import (
    fetch_external_data,
    fill_context_for_report,
    get_current_month,
    get_user_id,
    get_user_location,
    get_weather,
    rag_summarize,
)
from model.factory import chat_model
from utils.logger_handler import logger
from utils.prompt_loader import load_report_prompts, load_system_prompts


REACT_FORMAT_INSTRUCTIONS = """

You can use the following tools:

{tools}

Use this format:

Question: the input question you must answer
Thought: think about what to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer to the original input

Begin.

Question: {input}
Thought:{agent_scratchpad}
"""


class ReactAgent:
    def __init__(self):
        self.tools = [
            rag_summarize,
            get_weather,
            get_user_location,
            get_user_id,
            get_current_month,
            fetch_external_data,
            fill_context_for_report,
        ]
        self.normal_agent = self._create_executor(load_system_prompts(), verbose=True)
        self.report_agent = self._create_executor(load_report_prompts(), verbose=True)
        self._normal_eval_agent = None
        self._report_eval_agent = None

    def _create_executor(
        self,
        system_prompt: str,
        *,
        verbose: bool = False,
        return_intermediate_steps: bool = False,
    ) -> AgentExecutor:
        prompt = PromptTemplate.from_template(system_prompt + REACT_FORMAT_INSTRUCTIONS)
        agent = create_react_agent(chat_model, self.tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            handle_parsing_errors=True,
            max_iterations=8,
            return_intermediate_steps=return_intermediate_steps,
            verbose=verbose,
        )

    @staticmethod
    def _is_report_query(query: str) -> bool:
        report_keywords = ("报告", "使用记录", "使用情况", "月报", "保养建议", "统计")
        return any(keyword in query for keyword in report_keywords)

    def _select_executor(self, query: str) -> AgentExecutor:
        if self._is_report_query(query):
            logger.info("[ReactAgent] use report prompt")
            return self.report_agent

        logger.info("[ReactAgent] use normal prompt")
        return self.normal_agent

    def _select_eval_executor(self, query: str) -> AgentExecutor:
        if self._is_report_query(query):
            if self._report_eval_agent is None:
                self._report_eval_agent = self._create_executor(
                    load_report_prompts(),
                    return_intermediate_steps=True,
                )
            return self._report_eval_agent

        if self._normal_eval_agent is None:
            self._normal_eval_agent = self._create_executor(
                load_system_prompts(),
                return_intermediate_steps=True,
            )
        return self._normal_eval_agent

    def evaluate_once(self, query: str) -> dict:
        executor = self._select_eval_executor(query)
        result = executor.invoke({"input": query})
        tools_used = [
            action.tool
            for action, _observation in result.get("intermediate_steps", [])
        ]
        return {
            "query": query,
            "answer": result.get("output", ""),
            "tools_used": tools_used,
            "route": "report" if self._is_report_query(query) else "normal",
        }

    def execute_stream(self, query: str):
        executor = self._select_executor(query)

        try:
            for chunk in executor.stream({"input": query}):
                output = chunk.get("output")
                if output:
                    yield output.strip() + "\n"
        except Exception:
            logger.exception("[ReactAgent] execution failed")
            raise


if __name__ == "__main__":
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
