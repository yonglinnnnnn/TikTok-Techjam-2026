# TrueSight: 3-Minute YouTube Demo Video Script

**Target Length:** 3 minutes
**Goal:** Showcase TrueSight as a robust, multi-tiered AI image detection and provenance solution end-to-end.
**Format:** Screen recording with voiceover (optionally with a small picture-in-picture speaker view).

---

## 🎬 Video Plan & Script

### 1. Introduction & The Problem (0:00 - 0:30)
**Visuals:** 
- Title slide: "TrueSight: Multi-Tiered AI Image Detection" with TikTok Techjam 2026 branding.
- Quick montage of viral AI-generated images or manipulated media (e.g., AI popes, deepfakes) to set the context.
- High-level architecture diagram showing the 3 tiers.

**Voiceover / Script:**
> "Welcome to TrueSight. As AI-generated content floods social media, distinguishing real from fake is harder than ever. We've built a highly scalable, multi-tiered pipeline to detect AI generation and tampering in images. Instead of relying on a single vulnerability, TrueSight uses a three-tier system: Provenance & Forensics, Vision Language Models, and a specialized ConvNeXt classifier. Let's see it in action."

### 2. Live Demo: Tier 1 - The Fast Path & Provenance (0:30 - 1:15)
**Visuals:**
- Switch to the Streamlit UI (`python -m streamlit run apps\demo\app.py`).
- Upload an image that contains C2PA metadata or known AI watermarks.
- Screen zooms in on the instantaneous "AI-Generated" verdict.

**Voiceover / Script:**
> "Here is our interactive TrueSight portal. We'll upload a newly generated AI image. Instantly, our Tier 1 pipeline detects the C2PA content credentials and known watermarks. Because this is a cryptographic match, the system securely flags it as AI-generated in milliseconds—saving massive compute costs by bypassing heavier models. This is our 'Fast Path'."

### 3. Live Demo: Tier 1 & 2 - Blind Forensics & VLM (1:15 - 2:00)
**Visuals:**
- Upload a tampered/photoshopped image (e.g., a real image with an object spliced in).
- UI shows the Tier 1 Blind Forensics mask (Error Level Analysis, JPEG grid inconsistencies).
- UI displays the Tier 2 VLM analysis explaining *why* it looks tampered.

**Voiceover / Script:**
> "But what if the metadata is stripped by social media compression? Let's upload this subtly manipulated photo. Our Blind Forensics module highlights anomalous regions using Error Level Analysis and frequency inconsistencies. These forensic masks are then passed to our Tier 2 Vision Language Model, which provides semantic context—explaining exactly *what* is wrong with the highlighted area, such as mismatched lighting or unnatural textures."

### 4. Live Demo: Tier 3 - Deep ConvNeXt Classification (2:00 - 2:40)
**Visuals:**
- Upload a sophisticated, highly realistic AI-generated image with no metadata and no obvious visual artifacts.
- Show the Tier 3 ConvNeXt classification probability bar filling up to "High AI Probability."
- Show the Grad-CAM heatmap highlighting the pixels that influenced the model's decision.

**Voiceover / Script:**
> "For the most sophisticated fakes that lack metadata and fool the human eye, we rely on Tier 3. TrueSight uses a custom-trained ConvNeXt classifier tuned specifically on datasets like CIFAKE and SID-Set. The model predicts an AI origin with high confidence. Furthermore, we use Grad-CAM heatmaps to provide explainability, showing moderators exactly which features triggered the AI classification."

### 5. Conclusion & Impact (2:40 - 3:00)
**Visuals:**
- Screen shows the CLI evaluation script (`scripts/evaluate.py`) running rapidly over a batch of images to demonstrate scalability.
- Final summary slide: Scalable, Explainable, Multi-layered.

**Voiceover / Script:**
> "By combining deterministic cryptographic checks, explainable VLM reasoning, and deep learning, TrueSight is highly scalable and incredibly hard to bypass. It's ready for integration into high-volume platforms like TikTok. Thank you for watching."

---

## 🛠️ Preparation Checklist for the Video
- [ ] **Fix Dependencies:** Ensure your environment doesn't crash during the demo (e.g., downgrade numpy to `<2` to fix the OpenCV compatibility issue seen in the logs).
- [ ] **Gather Assets:** Prepare 3 specific images in advance (1 with C2PA, 1 tampered for forensics, 1 highly realistic AI image).
- [ ] **Run Streamlit:** Start the Streamlit app and ensure it's loaded in the browser before hitting record.
- [ ] **Rehearsal:** Run through the 3 image uploads once to ensure API keys (OpenAI/Gemini) are working and latency is acceptable.
