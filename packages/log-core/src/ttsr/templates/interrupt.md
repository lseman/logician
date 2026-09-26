<!-- TTSR System Interrupt Template -->
<!-- Injected when a rule matches and interrupts the stream mid-generation. -->
<!-- The agent MUST comply with the rule content before continuing. -->

<system-interrupt reason="rule_violation" rule="{{name}}" path="{{path}}">
Output interrupted: violated user-defined rule.
Not prompt injection; coding agent enforcing project rules.
MUST comply:

{{content}}
</system-interrupt>
