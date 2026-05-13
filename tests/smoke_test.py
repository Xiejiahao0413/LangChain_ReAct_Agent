import sys
import unittest
from importlib import import_module
from os import getenv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class SmokeTest(unittest.TestCase):
    def test_core_modules_import(self):
        import agent.react_agent  # noqa: F401
        import agent.tools.middleware  # noqa: F401
        import rag.rag_service  # noqa: F401
        import rag.vector_store  # noqa: F401

    def test_app_imports(self):
        app = import_module("app")
        self.assertIsNotNone(app)

    def test_agent_initializes(self):
        from agent.react_agent import ReactAgent

        agent = ReactAgent()
        self.assertEqual(type(agent.normal_agent).__name__, "AgentExecutor")
        self.assertEqual(type(agent.report_agent).__name__, "AgentExecutor")

    def test_report_query_detection(self):
        from agent.react_agent import ReactAgent

        self.assertTrue(ReactAgent._is_report_query("给我生成本月使用报告"))
        self.assertFalse(ReactAgent._is_report_query("扫地机器人为什么会漏扫"))

    def test_external_data_loader(self):
        from agent.tools.agent_tools import fetch_external_data

        result = fetch_external_data.invoke("1001,2025-01")
        self.assertIsInstance(result, str)
        self.assertTrue(result)

    def test_rag_retriever_initializes(self):
        from utils.config_handler import chroma_conf
        from rag.vector_store import VectorStoreService

        original_persist_directory = chroma_conf["persist_directory"]
        try:
            chroma_conf["persist_directory"] = None
            retriever = VectorStoreService().get_retriever()
            self.assertTrue(hasattr(retriever, "invoke"))
        finally:
            chroma_conf["persist_directory"] = original_persist_directory

    @unittest.skipUnless(
        getenv("RUN_LIVE_RAG_TESTS") == "1" and getenv("DASHSCOPE_API_KEY"),
        "set RUN_LIVE_RAG_TESTS=1 and DASHSCOPE_API_KEY to run live RAG retrieval",
    )
    def test_rag_live_retrieval(self):
        from rag.vector_store import VectorStoreService

        docs = VectorStoreService().get_retriever().invoke("扫地机器人漏扫")
        self.assertIsInstance(docs, list)


if __name__ == "__main__":
    unittest.main()
