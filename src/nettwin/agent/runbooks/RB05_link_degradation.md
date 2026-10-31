# RB05 Link degradation (loss or latency)

Fault type: link_degradation. Root: a link.

Symptoms:
- The link stays up, but its one-hop latency rises (link_latency_high) or it drops packets (link_discards).
- Services crossing the link breach latency or loss SLAs; services not crossing it are fine.
- Utilisation is often unchanged, because routing does not move traffic away from a degraded but up link.

Checks:
- Compare the link's latency and discards with the same window one day earlier.
- Find the element shared by the paths of the services with SLA alerts; a degraded link is usually on all of them.
- Small degradations may raise no alert: look at the telemetry of links on the shared paths even without alerts.

Action: raise with the transmission team; consider moving traffic off the link with a metric change.
