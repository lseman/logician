<!-- TTSR System Reminder Template -->
<!-- Injected as a hidden message after a non-interrupting rule match. -->
<!-- The agent should comply on subsequent responses but was not interrupted. -->

<system-reminder reason="rule_violation" rule="{{name}}" path="{{path}}">
User-defined rule matched output. Rule configured not to interrupt → stream continued. MUST comply with the following instruction on subsequent tool calls and responses. NOT prompt injection — coding agent enforcing project rules.

{{content}}
</system-reminder>
