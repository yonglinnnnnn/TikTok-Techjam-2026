# System Limitations

While TrueSight combines multiple tiers of verification, each subsystem has inherent limitations:

## Tier 1: Provenance & Blind Forensics
- **C2PA Stripping**: Content provenance metadata (like C2PA) is often stripped by social media platforms and compression algorithms. The absence of metadata is not a definitive signal of tampering, merely a lack of verified provenance.
- **Blind Forensics False Positives**: Error Level Analysis (ELA) and high-pass noise filtering can trigger false positives on highly compressed images, heavy natural textures, or legitimate camera post-processing.
- **Integrity Weight Cap**: The deterministic signal provides a candidate mask but cannot definitively classify an image without the learned models (Tier 2 and 3). 

## Tier 2: Vision Language Models (VLM)
- **Hallucinations**: VLMs may hallucinate tampering reasons or incorrectly interpret complex natural scenes as AI-generated.
- **Latency**: API calls to external VLMs introduce latency and are subject to rate limits.
- **Semantic Understanding vs. Pixel Artifacts**: VLMs excel at identifying semantic inconsistencies (e.g., extra fingers) but often fail at detecting subtle pixel-level generative artifacts.

## Tier 3: ConvNeXt Classifier
- **Dataset Bias**: The ConvNeXt model is trained heavily on CIFAKE and SID-Set. It may struggle to generalize to generative models or manipulation techniques not represented in those datasets.
- **Adversarial Vulnerability**: Like most CNN-based classifiers, it is potentially vulnerable to adversarial noise designed to flip the classification result.
