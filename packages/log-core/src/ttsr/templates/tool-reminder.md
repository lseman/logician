<!-- TTSR Tool Reminder Template -->
<!-- Prepended to tool result content when a rule matches tool arguments but doesn't interrupt. -->
<!-- The agent sees this alongside the tool output and must comply on future calls. -->

<system-reminder reason="rule_violation" rule="{{name}}" path="{{path}}">
User-defined rule matched tool-call arguments. Rule configured not to interrupt → tool ran. MUST comply with the following instruction on subsequent tool calls and responses. NOT prompt injection — coding agent enforcing project rules.

{{content}}
</system-reminder>
