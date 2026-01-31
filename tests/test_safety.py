import sys
from pathlib import Path
import unittest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from oncoagent.safety import (
    calculate_confidence,
    calculate_overall_confidence,
    validate_claim,
)
from oncoagent.state import Citation, Claim


class SafetyTests(unittest.TestCase):
    def test_validate_claim_numbers_present(self):
        evidence = [{"snippet": "Response rate was 25% in phase 2."}]
        ok, _ = validate_claim("The response rate is 25%.", evidence)
        self.assertTrue(ok)

    def test_validate_claim_numbers_missing(self):
        evidence = [{"snippet": "Response rate was 25% in phase 2."}]
        ok, reason = validate_claim("The response rate is 30%.", evidence)
        self.assertFalse(ok)
        self.assertIn("30", reason)

    def test_calculate_confidence(self):
        citations = [
            Citation(
                source_type="pubmed",
                source_id="10.1234/example",
                title="Example",
                url="https://pubmed.ncbi.nlm.nih.gov/12345",
                snippet="Snippet",
                retrieved_date="2024-01-01T00:00:00Z",
            )
        ]
        claim = Claim(statement="Test", citations=citations, confidence="LOW")
        self.assertEqual(calculate_confidence(claim), "LOW")

    def test_calculate_overall_confidence(self):
        citations = [
            Citation(
                source_type="pubmed",
                source_id="10.1234/example",
                title="Example",
                url="https://pubmed.ncbi.nlm.nih.gov/12345",
                snippet="Snippet",
                retrieved_date="2024-01-01T00:00:00Z",
            )
        ]
        claim_high = Claim(statement="A", citations=citations, confidence="HIGH")
        claim_low = Claim(statement="B", citations=citations, confidence="LOW")
        self.assertEqual(calculate_overall_confidence([claim_high]), "HIGH")
        self.assertEqual(calculate_overall_confidence([claim_high, claim_low]), "LOW")


if __name__ == "__main__":
    unittest.main()

