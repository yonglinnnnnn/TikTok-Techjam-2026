# TrueSight architecture

Tier 1 reads original bytes and produces provenance, normalized provider signals,
weak blind-forensic localization, routing, and a severity contribution. It never
re-encodes the source before provenance checks.

C2PA integrity and signer trust remain distinct. A valid-but-untrusted credential
is retained as `detected`; only a credential whose signer is trusted can drive a
verified AI or verified capture result. Provenance severity measures AI evidence,
while blind-forensic integrity risk is carried separately.

If Tier 1 has no verified AI signal, Tier 2 and Tier 3 operate on a separate
normalized image. Tier 2 performs semantic/VLM analysis and estimates AI coverage.
Tier 3 produces an independent ConvNeXt probability and Grad-CAM explanation.
The fusion layer owns the final verdict and calibrated confidence.

Interfaces and ownership are defined in `docs/team_integration.md`; the machine-
readable contract is `schemas/prediction.schema.json`.
