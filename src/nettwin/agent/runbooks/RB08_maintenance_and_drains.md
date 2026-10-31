# RB08 Planned maintenance and drains (general)

Applies to: incidents that overlap a change-log entry.

- A drain raises the routing metric of a link so traffic leaves it before work starts. The drained link's utilisation falls; neighbours carry more. Latency on rerouted services can rise. Nothing is broken.
- A planned outage ("down") takes a link or router down on purpose. Telemetry on it stops or reads zero, exactly as in a real failure, and status alerts fire.
- A no-op entry is logged work that did not touch traffic.
- During and just after a change window, alerts on the changed element and on links that took its traffic are expected.

How to decide:
- If every alerted element is the changed element or carries its rerouted traffic, and nothing else is abnormal, the answer is planned maintenance, not a fault.
- A fault can still happen during maintenance: look for symptoms on elements the change does not explain.
- Do not blame the changed element for symptoms that started before the change window.
