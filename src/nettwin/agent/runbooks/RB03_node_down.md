# RB03 Router (PoP) down

Fault type: node_down. Root: a PoP (router).

Symptoms:
- The router stops answering polls (pop_unreachable) when status polling exists.
- Every link attached to the router goes dark or reads zero at the same time.
- Services that start or end at the PoP become unreachable; services that transited it reroute.

Checks:
- Do all links of one PoP show the same outage at the same step? A single dark link is a link fault; all links of one router is the router.
- Is there a planned outage on the PoP in the change log?

Action: dispatch to the site; check power and control plane; confirm traffic has moved to alternate paths.
