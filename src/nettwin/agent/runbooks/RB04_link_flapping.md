# RB04 Link flapping

Fault type: link_flapping. Root: a link.

Symptoms:
- The interface goes down and up repeatedly; several short link_status_down alerts on the same link, or intermittent gaps in its telemetry.
- Repeated short bursts of loss on services that cross the link as routing converges each time.
- Utilisation on the link drops and recovers repeatedly; neighbours show matching opposite swings.

Checks:
- Count down/up episodes on the link in the window: more than one short episode is the signature.
- Rule out a collector problem: intermittent gaps on many unrelated links at once are not flapping.

Action: dampen or shut the interface and reroute; check optics and physical layer.
