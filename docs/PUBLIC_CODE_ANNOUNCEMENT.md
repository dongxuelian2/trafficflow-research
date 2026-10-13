Title: Independent four-task research code and reproducible validation experiments

We are sharing our independent TrafficFlowBench research implementation:

https://github.com/dongxuelian2/trafficflow-research

The repository contains data validation, the four official baseline integrations,
our own State/Queue/OD models, strict complete-submission validation, and research
experiments using only the current public release. It also includes an independent
CTM stress simulator, event-based observation-proxy Queue validation, and known-truth
synthetic OD comparisons with hierarchical probabilistic priors.

The first verified public leaderboard submission scored 0.82940. Local State
scores, noisy-observation Queue proxies, simulated Physics metrics and simulated
OD scores are explicitly separate from the formal competition score. Negative
results and reproducibility limitations are recorded alongside successful ones.

We implement our methods independently and do not use hidden labels, organizer-only
assets, another team's predictions, or the public reference solve as path truth.
The repository is MIT licensed. Data and large local model artifacts are not
redistributed; please obtain the current public data from the competition Data page.

Research is ongoing. The current results and reproducible commands are documented
in RESULTS.md, EXPERIMENTS.md, MODEL_COMPARISON.md and RESEARCH_ROADMAP.md.
