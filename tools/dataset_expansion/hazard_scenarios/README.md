# SafeBand AI — Hazard Scenario Generation

Stage: 2.10.7

Purpose:
Generate explicitly labeled synthetic hazard scenarios from real sensor sessions.

Profiles:

- FALL_FORWARD
- FALL_BACKWARD
- FALL_SIDE
- TRIP_RECOVERY
- SUDDEN_COLLAPSE
- HIGH_IMPACT
- LOSS_OF_BALANCE

Input:
Real sensor session.

Output:
Original sensor columns plus:

- hazard_label
- hazard_type
- hazard_event_id
- is_hazard

Hazard rows are labeled 1.
Non-hazard rows are labeled 0.

Status:
SYNTHETIC_UNVALIDATED

The generated scenarios are development/training fixtures and are not
claims of real-world fall-detection performance.

Preserved properties:

- source row count
- sample_index
- activity
- elapsed_ms
- validity flags
- sensor sentinel semantics
- original row ordering

Modified properties:

- accelerometer values
- gyroscope values
- explicit hazard metadata

Each generation uses a deterministic seed and produces a provenance JSON file.
