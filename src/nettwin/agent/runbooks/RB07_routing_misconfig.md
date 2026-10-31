# RB07 Routing misconfiguration (link metric change)

Fault type: routing_misconfig. Root: the link whose routing metric was changed.

Symptoms:
- Traffic moves suddenly at one step: utilisation drops on one link and rises on others (metric raised), or rises sharply on one link (metric lowered, pulling traffic onto it).
- No interface goes down. Service latency changes because paths changed.
- Unlike a drain, there is no matching entry in the change log.

Checks:
- Look for a step change in utilisation on one link with no outage and no logged change.
- If a link lost most of its traffic while staying up, and nothing is logged, the metric on that link is the likely root.
- If one link suddenly carries much more than the day before, check whether paths were pulled onto it.

Action: compare the running routing metric with the intended configuration; roll back the change.
