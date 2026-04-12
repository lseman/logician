from __future__ import annotations

from src.tools.runtime import Tool, ToolParameter, validate_tool_arguments


def _echo_tool(**kwargs):
    return {"status": "ok", **kwargs}


def test_runtime_validator_remaps_aliases_and_coerces_types():
    tool = Tool(
        name="load_csv",
        description="Load a CSV file.",
        parameters=[
            ToolParameter(name="filepath", type="string", description="CSV path."),
            ToolParameter(name="limit", type="integer", description="Row limit.", required=False),
        ],
        function=_echo_tool,
    )

    prepared, error = validate_tool_arguments(
        tool,
        {"path": "demo.csv", "limit": "3"},
        common_aliases={"filepath": {"path", "file", "filename"}},
    )

    assert error is None
    assert prepared == {"filepath": "demo.csv", "limit": 3}


def test_runtime_validator_reports_missing_and_unknown_fields():
    tool = Tool(
        name="echo_pair",
        description="Echo two values.",
        parameters=[
            ToolParameter(name="left", type="string", description="Left value."),
            ToolParameter(name="right", type="string", description="Right value."),
        ],
        function=_echo_tool,
    )

    prepared, error = validate_tool_arguments(tool, {"left": "A", "extra": "C"})

    assert prepared is None
    assert error is not None
    payload = error.to_payload()
    assert payload["error_type"] == "schema_validation_failed"
    assert payload["missing_required"] == ["right"]
    assert payload["unknown_arguments"] == ["extra"]


def test_runtime_validator_reports_enum_type_errors():
    tool = Tool(
        name="choose_mode",
        description="Choose an execution mode.",
        parameters=[
            ToolParameter(
                name="mode",
                type="string",
                description="Execution mode.",
                enum=["fast", "safe"],
            )
        ],
        function=_echo_tool,
    )

    prepared, error = validate_tool_arguments(tool, {"mode": "turbo"})

    assert prepared is None
    assert error is not None
    payload = error.to_payload()
    assert payload["error_type"] == "schema_type_validation_failed"
    assert payload["type_errors"][0]["allowed_values"] == ["fast", "safe"]
