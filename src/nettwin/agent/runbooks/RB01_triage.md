# RB01 Incident triage (general)

Applies to: every incident. Fault types: all.

1. Read the alert list for the window. Group alerts by element. Service alerts (svc_loss_sla, svc_latency_sla, svc_unreachable) are symptoms seen at the edge; link and router alerts point closer to the cause.
2. Check the change log before anything else. A planned drain, outage or other change on an element in or just before the window explains alerts on that element and on links that took its rerouted traffic.
3. For each candidate element, look at its telemetry in the window against the same window one day earlier. A real fault changes something that was normal the day before.
4. Use the topology: a single root usually explains several symptoms. Find the element shared by the paths of the alerted services (shared elements), and check its neighbours.
5. Decide between: a fault (name the root element and type), planned maintenance only, or no fault. Do not report a fault when every symptom is explained by a logged change.
6. Evidence must quote element ids and values exactly as the tools returned them.
