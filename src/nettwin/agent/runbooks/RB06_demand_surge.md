# RB06 Demand surge at a PoP

Fault type: demand_surge. Root: a PoP (the source of the extra traffic).

Symptoms:
- Utilisation rises together on several links attached to one PoP, ramping up over some steps rather than jumping.
- Downstream links on paths from that PoP may hit high utilisation (link_util_high) and start to discard.
- Latency and loss rise on services from that PoP once links congest.

Checks:
- Do the busiest links share one end router? The shared router is the root, not the busiest single link.
- Is the rise larger than the normal daily pattern? Compare with the same window one day earlier.
- A drain in the change log also raises utilisation on neighbouring links; check that first.

Action: identify the traffic source; apply rate limits or add capacity; reroute where possible.
