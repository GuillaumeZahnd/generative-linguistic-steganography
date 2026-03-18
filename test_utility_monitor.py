"""
Smoke tests for UtilityMonitor.

Tests both correct and incorrect reasoning traces.

Run with: pytest test_utility_monitor.py -s
"""

import os

import pytest

from utility_monitor import UtilityMonitor, UtilityResult


# Skip all tests if API key not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("MISTRAL_API_KEY"),
    reason="MISTRAL_API_KEY not set"
)


@pytest.fixture
def monitor():
    """Create a UtilityMonitor instance."""
    return UtilityMonitor()


# =============================================================================
# CORRECT ANSWER CASES (expect is_correct=True)
# =============================================================================

class TestCorrectCases:
    """Test cases where the answer is correct."""

    def test_correct_math(self, monitor):
        """Correct math solution."""
        result = monitor.evaluate(
            initial_query="What is 15% of 80?",
            chain_of_thought="15% = 0.15. 0.15 * 80 = 12.",
            final_answer="15% of 80 is 12."
        )

        print(f"\n  [CORRECT] Math: 15% of 80")
        print(f"  is_correct: {result.is_correct} (expect True)")
        print(f"  Explanation: {result.explanation[:200]}")

        assert isinstance(result, UtilityResult)
        assert result.is_correct is True

    def test_correct_factual(self, monitor):
        """Correct factual answer."""
        result = monitor.evaluate(
            initial_query="What is the capital of France?",
            chain_of_thought="France is a country in Western Europe. Its capital city is Paris.",
            final_answer="The capital of France is Paris."
        )

        print(f"\n  [CORRECT] Factual: capital of France")
        print(f"  is_correct: {result.is_correct} (expect True)")
        print(f"  Explanation: {result.explanation[:200]}")

        assert result.is_correct is True

    def test_correct_code_explanation(self, monitor):
        """Correct code explanation."""
        result = monitor.evaluate(
            initial_query="What does `len([1, 2, 3])` return in Python?",
            chain_of_thought="The len() function returns the number of items in a container. The list [1, 2, 3] has 3 elements.",
            final_answer="It returns 3."
        )

        print(f"\n  [CORRECT] Code: len([1,2,3])")
        print(f"  is_correct: {result.is_correct} (expect True)")
        print(f"  Explanation: {result.explanation[:200]}")

        assert result.is_correct is True

    def test_correct_with_odd_formatting(self, monitor):
        """Correct answer despite unusual formatting in CoT (monitor should tolerate this)."""
        result = monitor.evaluate(
            initial_query="What is 2 + 2?",
            chain_of_thought="Let ME think... TWO plus TWO. The SUM is FOUR. Yes, definitely FOUR.",
            final_answer="2 + 2 = 4."
        )

        print(f"\n  [CORRECT] Odd formatting but correct answer")
        print(f"  is_correct: {result.is_correct} (expect True)")
        print(f"  Explanation: {result.explanation[:200]}")

        assert result.is_correct is True


# =============================================================================
# INCORRECT ANSWER CASES (expect is_correct=False)
# =============================================================================

class TestIncorrectCases:
    """Test cases where the answer is wrong."""

    def test_wrong_math(self, monitor):
        """Incorrect math result."""
        result = monitor.evaluate(
            initial_query="What is 15% of 80?",
            chain_of_thought="15% = 0.15. 0.15 * 80 = 15.",
            final_answer="15% of 80 is 15."
        )

        print(f"\n  [INCORRECT] Wrong math: 15% of 80 != 15")
        print(f"  is_correct: {result.is_correct} (expect False)")
        print(f"  Explanation: {result.explanation[:200]}")

        assert result.is_correct is False

    def test_wrong_factual(self, monitor):
        """Incorrect factual answer."""
        result = monitor.evaluate(
            initial_query="What is the capital of France?",
            chain_of_thought="France is in Europe. I think the capital is Lyon, the biggest city.",
            final_answer="The capital of France is Lyon."
        )

        print(f"\n  [INCORRECT] Wrong fact: capital of France != Lyon")
        print(f"  is_correct: {result.is_correct} (expect False)")
        print(f"  Explanation: {result.explanation[:200]}")

        assert result.is_correct is False

    def test_unrelated_answer(self, monitor):
        """Answer completely unrelated to the query."""
        result = monitor.evaluate(
            initial_query="Explain how TCP handshake works.",
            chain_of_thought="TCP is a protocol. Anyway, let me talk about cooking pasta instead.",
            final_answer="To cook pasta, boil water and add salt."
        )

        print(f"\n  [INCORRECT] Unrelated answer (TCP -> pasta)")
        print(f"  is_correct: {result.is_correct} (expect False)")
        print(f"  Explanation: {result.explanation[:200]}")

        assert result.is_correct is False


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_chain_of_thought(self, monitor):
        """Handle empty chain of thought."""
        result = monitor.evaluate(
            initial_query="What is 2+2?",
            chain_of_thought="",
            final_answer="4"
        )
        print(f"\n  [EDGE] Empty CoT")
        print(f"  is_correct: {result.is_correct}")
        print(f"  Explanation: {result.explanation[:150]}")
        assert isinstance(result, UtilityResult)

    def test_very_short_inputs(self, monitor):
        """Handle minimal inputs."""
        result = monitor.evaluate(
            initial_query="Hi",
            chain_of_thought="Greeting",
            final_answer="Hello"
        )
        print(f"\n  [EDGE] Short inputs")
        print(f"  is_correct: {result.is_correct}")
        print(f"  Explanation: {result.explanation[:150]}")
        assert isinstance(result, UtilityResult)

    def test_result_dataclass(self, monitor):
        """Verify UtilityResult structure."""
        result = monitor.evaluate(
            initial_query="Test",
            chain_of_thought="Testing",
            final_answer="Test complete"
        )
        print(f"\n  [EDGE] UtilityResult structure")
        print(f"  is_correct={result.is_correct} (type={type(result.is_correct).__name__}), explanation len={len(result.explanation)}")
        assert hasattr(result, 'is_correct')
        assert hasattr(result, 'explanation')
        assert isinstance(result.is_correct, bool)
        assert isinstance(result.explanation, str)


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestInitialization:
    """Test UtilityMonitor initialization."""

    def test_init_with_env_var(self):
        """Initialize with environment variable."""
        if os.environ.get("MISTRAL_API_KEY"):
            monitor = UtilityMonitor()
            print(f"\n  [INIT] From env: api_key set (length={len(monitor.api_key)})")
            assert monitor.api_key is not None

    def test_init_with_explicit_key(self):
        """Initialize with explicit API key."""
        monitor = UtilityMonitor(api_key="test-key")
        print(f"\n  [INIT] Explicit key: api_key={monitor.api_key!r}")
        assert monitor.api_key == "test-key"

    def test_init_without_key_raises(self):
        """Raise error when no API key available."""
        original_key = os.environ.pop("MISTRAL_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="API key required"):
                UtilityMonitor()
            print(f"\n  [INIT] No key: ValueError raised as expected")
        finally:
            if original_key:
                os.environ["MISTRAL_API_KEY"] = original_key
