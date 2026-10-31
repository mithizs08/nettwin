# RB09 Telemetry gaps and collector problems (general)

Applies to: alerts of type link_no_data or link_probe_fail, and missing values in series.

- Missing telemetry is not a fault by itself. Collectors drop data for single links or for every link of one router for a few steps.
- A down link also stops reporting. Tell them apart:
  - A collector gap is usually short and often hits several unrelated links or a whole router at once, while services crossing them are fine.
  - When a link goes down, its traffic moves to neighbours (their utilisation rises), and services crossing it see loss at the start.
- A missing one-hop latency probe while counters still report means the probe failed, not the link.
- When status polling is not available, telemetry absence plus traffic moving to neighbours is the main evidence of an outage.
