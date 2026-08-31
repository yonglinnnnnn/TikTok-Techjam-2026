# TrueSight architecture

Tier 1 reads original bytes and produces provenance, normalized provider signals,
weak blind-forensic localization, routing, and a severity contribution. It never
re-encodes the source before provenance checks.

If Tier 1 has no verified AI signal, Tier 2 and Tier 3 operate on a separate
normalized image. Tier 2 performs semantic/VLM analysis and estimates AI coverage.
Tier 3 produces an independent ConvNeXt probability and Grad-CAM explanation.
The fusion layer owns the final verdict and calibrated confidence.

Interfaces and ownership are defined in `docs/team_integration.md`; the machine-
readable contract is `schemas/prediction.schema.json`.
