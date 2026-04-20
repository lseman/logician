"""Tests for src/agent/classify.py — classify_turn() function."""

from __future__ import annotations

from src.agent.classify import TurnClassification, classify_turn


class TestTurnClassification:
    """Test TurnClassification dataclass."""

    def test_turn_classification_basic(self) -> None:
        """TurnClassification stores intent and domain_groups."""
        tc = TurnClassification(intent="execution", domain_groups={"timeseries"})
        assert tc.intent == "execution"
        assert tc.domain_groups == {"timeseries"}

    def test_turn_classification_multiple_domains(self) -> None:
        """TurnClassification can hold multiple domain groups."""
        tc = TurnClassification(
            intent="design", domain_groups={"timeseries", "academic", "svg"}
        )
        assert tc.intent == "design"
        assert tc.domain_groups == {"timeseries", "academic", "svg"}


class TestClassifyTurn:
    """Test classify_turn() function."""

    # Intent detection tests

    def test_social_intent_hello(self) -> None:
        """Detect 'hello' as social intent."""
        result = classify_turn("hello, how can I help?")
        assert result.intent == "social"

    def test_social_intent_hi(self) -> None:
        """Detect 'hi ' as social intent."""
        result = classify_turn("hi there!")
        assert result.intent == "social"

    def test_social_intent_thanks(self) -> None:
        """Detect 'thanks' as social intent."""
        result = classify_turn("thanks for the help")
        assert result.intent == "social"

    def test_social_intent_thank_you(self) -> None:
        """Detect 'thank you' as social intent."""
        result = classify_turn("thank you so much")
        assert result.intent == "social"

    def test_social_intent_good_morning(self) -> None:
        """Detect 'good morning' as social intent."""
        result = classify_turn("good morning! How are you today?")
        assert result.intent == "social"

    def test_social_intent_whats_up(self) -> None:
        """Detect 'what's up' as social intent."""
        result = classify_turn("what's up?")
        assert result.intent == "social"

    def test_design_intent_design(self) -> None:
        """Detect 'design ' as design intent."""
        result = classify_turn("design a new system for this")
        assert result.intent == "design"

    def test_design_intent_architect(self) -> None:
        """Detect 'architect' as design intent."""
        result = classify_turn("architect the solution differently")
        assert result.intent == "design"

    def test_design_intent_how_should_structure(self) -> None:
        """Detect 'how should i structure' as design intent."""
        result = classify_turn("how should i structure this codebase?")
        assert result.intent == "design"

    def test_design_intent_propose(self) -> None:
        """Detect 'propose ' as design intent."""
        result = classify_turn("propose a better approach")
        assert result.intent == "design"

    def test_design_intent_tradeoff(self) -> None:
        """Detect 'trade-off' as design intent."""
        result = classify_turn("what's the trade-off between A and B?")
        assert result.intent == "design"

    def test_informational_intent_explain(self) -> None:
        """Detect 'explain ' as informational intent."""
        result = classify_turn("explain how this works")
        assert result.intent == "informational"

    def test_informational_intent_what_is(self) -> None:
        """Detect 'what is ' as informational intent."""
        result = classify_turn("what is machine learning?")
        assert result.intent == "informational"

    def test_informational_intent_how_does(self) -> None:
        """Detect 'how does ' as informational intent."""
        result = classify_turn("how does the algorithm work?")
        assert result.intent == "informational"

    def test_informational_intent_describe(self) -> None:
        """Detect 'describe ' as informational intent."""
        result = classify_turn("describe the process")
        assert result.intent == "informational"

    def test_informational_intent_why_is(self) -> None:
        """Detect 'why is ' as informational intent."""
        result = classify_turn("why is this approach better?")
        assert result.intent == "informational"

    def test_execution_intent_default(self) -> None:
        """Default to 'execution' when no keyword matches."""
        result = classify_turn("run the tests")
        assert result.intent == "execution"

    def test_execution_intent_empty_string(self) -> None:
        """Empty string defaults to 'execution'."""
        result = classify_turn("")
        assert result.intent == "execution"

    def test_intent_first_match_wins(self) -> None:
        """First matching intent wins (order matters)."""
        # 'hello' is social, but if something else matches social before social is checked,
        # the result should still be social. But within our patterns, social comes first.
        result = classify_turn(
            "hello, explain how the system works"
        )  # Contains both social and informational
        assert result.intent == "social"  # social patterns checked first

    # Domain detection tests

    def test_domain_timeseries_reservoir(self) -> None:
        """Detect 'reservoir' as timeseries domain."""
        result = classify_turn("analyze the reservoir data")
        assert "timeseries" in result.domain_groups

    def test_domain_timeseries_forecast(self) -> None:
        """Detect 'forecast' as timeseries domain."""
        result = classify_turn("create a forecast model")
        assert "timeseries" in result.domain_groups

    def test_domain_timeseries_ons(self) -> None:
        """Detect 'ons' as timeseries domain."""
        result = classify_turn("ONS data analysis")
        assert "timeseries" in result.domain_groups

    def test_domain_timeseries_energy_data(self) -> None:
        """Detect 'energy data' as timeseries domain."""
        result = classify_turn("load the energy data")
        assert "timeseries" in result.domain_groups

    def test_domain_timeseries_hydroelectric(self) -> None:
        """Detect 'hydroelectric' as timeseries domain."""
        result = classify_turn("hydroelectric reservoir levels")
        assert "timeseries" in result.domain_groups

    def test_domain_academic_paper(self) -> None:
        """Detect 'paper' as academic domain."""
        result = classify_turn("review this paper")
        assert "academic" in result.domain_groups

    def test_domain_academic_citation(self) -> None:
        """Detect 'citation' as academic domain."""
        result = classify_turn("find citation information")
        assert "academic" in result.domain_groups

    def test_domain_academic_s2(self) -> None:
        """Detect 's2 ' as academic domain."""
        result = classify_turn("search s2 for papers")
        assert "academic" in result.domain_groups

    def test_domain_academic_ieee(self) -> None:
        """Detect 'ieee' as academic domain."""
        result = classify_turn("IEEE conference papers")
        assert "academic" in result.domain_groups

    def test_domain_academic_literature(self) -> None:
        """Detect 'literature' as academic domain."""
        result = classify_turn("review the literature")
        assert "academic" in result.domain_groups

    def test_domain_rag_ingest(self) -> None:
        """Detect 'ingest' as rag domain."""
        result = classify_turn("ingest the documents")
        assert "rag" in result.domain_groups

    def test_domain_rag_retrieve(self) -> None:
        """Detect 'retrieve' as rag domain."""
        result = classify_turn("retrieve relevant documents")
        assert "rag" in result.domain_groups

    def test_domain_rag_embed(self) -> None:
        """Detect 'embed' as rag domain."""
        result = classify_turn("embed the text")
        assert "rag" in result.domain_groups

    def test_domain_rag_vector_store(self) -> None:
        """Detect 'vector store' as rag domain."""
        result = classify_turn("query the vector store")
        assert "rag" in result.domain_groups

    def test_domain_rag_chromadb(self) -> None:
        """Detect 'chromadb' as rag domain."""
        result = classify_turn("chromadb operations")
        assert "rag" in result.domain_groups

    def test_domain_svg_svg(self) -> None:
        """Detect ' svg' as svg domain."""
        result = classify_turn("create an svg diagram")
        assert "svg" in result.domain_groups

    def test_domain_svg_diagram(self) -> None:
        """Detect 'diagram' as svg domain."""
        result = classify_turn("draw a diagram")
        assert "svg" in result.domain_groups

    def test_domain_svg_chart(self) -> None:
        """Detect ' chart' as svg domain."""
        result = classify_turn("create a chart visualization")
        assert "svg" in result.domain_groups

    def test_domain_svg_visualize(self) -> None:
        """Detect 'visualize' as svg domain."""
        result = classify_turn("visualize the data")
        assert "svg" in result.domain_groups

    # Multiple domains test

    def test_multiple_domains(self) -> None:
        """Detect multiple domains in one message."""
        result = classify_turn(
            "analyze the timeseries reservoir data and create a chart visualization"
        )
        assert "timeseries" in result.domain_groups
        assert "svg" in result.domain_groups
        assert len(result.domain_groups) >= 2

    def test_multiple_domains_timeseries_academic(self) -> None:
        """Detect both timeseries and academic domains."""
        result = classify_turn(
            "write a paper about the ONS hydroelectric forecast literature"
        )
        assert "timeseries" in result.domain_groups
        assert "academic" in result.domain_groups

    def test_multiple_domains_all_four(self) -> None:
        """Detect all four domains in one message."""
        result = classify_turn(
            "ingest energy data from ONS, embed citations, create an SVG chart for the paper"
        )
        assert "timeseries" in result.domain_groups
        assert "academic" in result.domain_groups
        assert "rag" in result.domain_groups
        assert "svg" in result.domain_groups

    # Edge cases

    def test_no_domains_detected(self) -> None:
        """Empty domain set when no domains match."""
        result = classify_turn("run the tests and verify")
        assert result.domain_groups == set()

    def test_empty_string_execution_no_domains(self) -> None:
        """Empty string → execution intent, empty domains."""
        result = classify_turn("")
        assert result.intent == "execution"
        assert result.domain_groups == set()

    def test_case_insensitive_matching(self) -> None:
        """Matching is case-insensitive."""
        result = classify_turn("HELLO, HOW ARE YOU?")
        assert result.intent == "social"

    def test_case_insensitive_domains(self) -> None:
        """Domain matching is case-insensitive."""
        result = classify_turn("FORECAST THE RESERVOIR DATA")
        assert "timeseries" in result.domain_groups

    def test_whitespace_handling(self) -> None:
        """Whitespace is preserved in lowercased content."""
        result = classify_turn("  hello  ")
        assert result.intent == "social"

    def test_partial_keyword_matching(self) -> None:
        """Keywords match as substrings."""
        result = classify_turn("this is helpful thanks a lot")
        assert result.intent == "social"  # contains "thanks"

    def test_multiple_intents_social_wins(self) -> None:
        """When multiple intent keywords present, first in pattern list wins."""
        # "hello" (social) comes before "how does" (informational) in patterns
        result = classify_turn("hello, how does this work?")
        assert result.intent == "social"

    def test_design_vs_execution(self) -> None:
        """Design intent takes precedence when keywords present."""
        result = classify_turn("design a new approach to execute queries")
        assert result.intent == "design"

    def test_result_type_is_turn_classification(self) -> None:
        """classify_turn returns TurnClassification instance."""
        result = classify_turn("some message")
        assert isinstance(result, TurnClassification)
        assert hasattr(result, "intent")
        assert hasattr(result, "domain_groups")
