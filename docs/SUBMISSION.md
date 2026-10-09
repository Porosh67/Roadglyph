# lablab.ai submission text (Mini Challenge 2)

Everything below is ready to paste. Limits checked: title 49/50 characters,
description 1305 characters (600-2000) and 203 words (min 100).

## Step 1 of 2: Basic Information

**Submission Title**

    RoadGlyph: Scene-Text OCR for Plates & Road Signs

**Long Description**

    RoadGlyph reads text from licence plates and road signs under real-world conditions: noise, blur, glare, low light and off-axis angles. It is a container for AMD GPUs on ROCm that runs the Qwen3-VL vision-language model through a resident server, so the model loads once instead of once per image.

    For every photo it measures brightness, glare, noise and sharpness, then builds enhanced variants such as brightened, de-glared or denoised copies. The model first classifies the object (plate, stop sign, speed limit, advisory plaque, work zone sign) and then transcribes each variant with a category-specific prompt. A deterministic post-processor enforces the challenge convention: state names and slogans are dropped from plates, Chinese province characters are kept, plaques return digits only, and multi-line signs are joined top to bottom. The variants vote using the grader's own normalisation, and two agreeing reads stop the run early.

    Reliability was a design goal. A per-image deadline and watchdog always write a valid JSON answer before the 30 second limit, VRAM is capped below 48 GiB, and PNG, JPEG, 16-bit and multi-frame TIFF inputs are all handled. The project ships with a Dockerfile built on the mandated ROCm base image, 63 unit tests, evaluation scripts and a pre-submission checker.

**Categories**: pick the closest of: Computer Vision / OCR / Machine Learning
(choose from the dropdown; whatever is offered).

**Technologies Used**: AMD ROCm, AMD Instinct / Developer Cloud, PyTorch, Hugging Face
Transformers, Qwen3-VL, Docker, Python (select the ones that exist in the list).

**Mini Challenge 2 Image**: in your screenshot this field contained the test text `xcv`.
Clear it. Put a real image URL or a screenshot only if the field accepts one, otherwise leave it empty.

**Mini Challenge 3 Image**: leave empty (you are not submitting Match 3 here).

## Step 2 of 2

Expected: the public container image reference. Push the image to a registry, make it public,
test `docker pull` without logging in, then paste the reference. Do not publish it anywhere else
(not in a public GitHub README, blog post or the description above).

## Before you press submit

- [ ] `./scripts/check_submission.sh <image>` passes
- [ ] real sample images from the brief were scored on the AMD GPU
- [ ] image is public and pullable without credentials
- [ ] no secrets in the image
