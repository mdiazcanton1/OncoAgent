import sys
from pathlib import Path
import unittest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from oncoagent.agents.supervisor import supervisor


class SupervisorTests(unittest.TestCase):
    def _base_state(self):
        return {
            "messages": [],
            "original_query": "test",
            "query_type": "general",
            "cancer_type": None,
            "images": [],
            "evidence": [],
            "clinical_trials": [],
            "claims": [],
            "current_agent": "",
            "agents_completed": [],
            "needs_cross_validation": False,
            "response": None,
            "confidence_overall": None,
        }

    def test_routes_to_vision_when_images_present(self):
        state = self._base_state()
        state["images"] = [{"data": b"fake", "mime_type": "image/jpeg"}]
        result = supervisor(state)
        self.assertEqual(result.goto, "vision")

    def test_routes_to_research_first(self):
        state = self._base_state()
        result = supervisor(state)
        self.assertEqual(result.goto, "research")


if __name__ == "__main__":
    unittest.main()

