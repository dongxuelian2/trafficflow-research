"""Shared independent CTM stress configurations, with no GPU dependency."""
CASES = {
    'mild': dict(capacity_retained=.85, demand_multiplier=.9, incident_cell=8, failure_fraction=0.),
    'onset': dict(capacity_retained=.5, demand_multiplier=1.1, incident_cell=20),
    'severe_upstream': dict(capacity_retained=.3, demand_multiplier=1.25, incident_cell=8),
    'severe_downstream': dict(capacity_retained=.3, demand_multiplier=1.25, incident_cell=27),
    'failures': dict(capacity_retained=.5, failure_fraction=.4, failure_steps=36),
    'noisy': dict(capacity_retained=.5, speed_noise=3., relative_flow_noise=.05)}
