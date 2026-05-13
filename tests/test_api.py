import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api.app import app, format_sse, get_agent  # noqa: E402


class FakeAgent:
    def execute_stream(self, query: str):
        yield f"收到：{query}"
        yield "\n建议先检查主刷和传感器。"


class ApiTest(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides[get_agent] = lambda: FakeAgent()
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_chat(self):
        response = self.client.post("/api/chat", json={"query": "漏扫怎么办"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["query"], "漏扫怎么办")
        self.assertIn("主刷", payload["answer"])

    def test_sse_stream_get(self):
        with self.client.stream("GET", "/api/chat/stream", params={"query": "生成报告"}) as response:
            self.assertEqual(response.status_code, 200)
            self.assertIn("text/event-stream", response.headers["content-type"])
            content = "".join(response.iter_text())
        self.assertIn("event: start", content)
        self.assertIn("event: message", content)
        self.assertIn("event: done", content)

    def test_format_sse(self):
        event = format_sse("message", {"delta": "你好"})
        self.assertEqual(event, 'event: message\ndata: {"delta": "你好"}\n\n')


if __name__ == "__main__":
    unittest.main()
