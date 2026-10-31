# RB02 Link down

Fault type: link_down. Root: a link.

Symptoms:
- The interface is reported down (link_status_down) when status polling exists.
- Telemetry on the link stops or reads zero: utilisation at 0 with the one-hop latency probe missing, or no data at all when status events are not available.
- Short loss on services that crossed the link while routing converges, then latency changes on the services that were rerouted.
- Neighbouring links (sharing a router) pick up the traffic: utilisation rises on them.

Checks:
- Is there a planned outage ("down") or drain on this link in the change log? If so it is maintenance, not a fault.
- Are many links blanked at the same time? That points to a collector gap (see RB09), not a link failure.

Action: open a fault ticket with the carrier or field team for the link; confirm rerouting has capacity.
