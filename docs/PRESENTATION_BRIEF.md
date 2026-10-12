# Presentation Brief: face-cloak

This brief contains everything needed to build the slide deck for **"Adversarial Image Cloaking for Protection Against Face-Swap Attacks"** without reading the code. All numbers come from the full experiment (`results/`, same numbers as `docs/REPORT.md`). Chart paths are relative to this `docs/` folder.

---

## 1. Deck metadata

| Item | Value |
|---|---|
| Audience | Course teacher / examiner and classmates (Multimedia Forensics viva), IIIT Kottayam |
| Length | 18 slides, about 20 minutes (speaker notes ≈ 18 min + 2–3 min live demo), plus Q&A |
| Presenters | Jacob Antony Jeejo, Manasa Manoj, Praneetha Reddy (equal contribution); suggested split: slides 1–6 / 7–13 / 14–18 |
| Report | `docs/REPORT.md` |
| Live site | https://face-cloak.vercel.app (Home, Live demo, Results, About) |
| Repo | https://github.com/J4KE-B/face-cloak |

**One-line pitch.** We add a barely visible, mathematically bounded perturbation to a photo so that face-swap tools receive the wrong identity: deepfakes made from the protected photo no longer match the person, as judged by a held-out face recogniser.

**Three key takeaways.**

1. One-shot, embedding-based swappers such as inswapper and SimSwap see a person's identity only through a face-recognition embedding. Attacking that embedding protects the photo, as judged by a held-out recogniser: PSR rises from 1.3 % to 99.3 % on inswapper (white-box encoder) and to 85.1 % on SimSwap (black-box), so 14.9 % of SimSwap swaps stay unprotected.
2. Expectation over Transformation (EOT) is the key ingredient: it makes the cloak survive JPEG 50 (98.7 % on inswapper) and lifts black-box transfer from 26.0 % to 85.1 %. An LPIPS term keeps the same protection at a much lower perceptual distortion (LPIPS 0.020 instead of 0.094).
3. It is not a silver bullet: strong blur (σ = 2) and bilateral filtering cut SimSwap protection to 46.4–48.7 % (the two EOT variants), and we did not test adaptive attackers or diffusion purification.

---

## 2. Slide-by-slide outline

Charts are in `../web/public/results/`. "PSR" = protection success rate = share of swaps that a held-out face recogniser no longer matches to the person.

### Slide 1. Title

- **On-slide:**
  - Adversarial Image Cloaking for Protection Against Face-Swap Attacks
  - Jacob Antony Jeejo · Manasa Manoj · Praneetha Reddy (equal contribution)
  - Multimedia Forensics · IIIT Kottayam
  - face-cloak.vercel.app
- **Speaker notes (about 30 s):** "Good morning. Our project is about protecting ordinary photos against face-swap deepfakes. Instead of detecting a deepfake after it is made, we change the photo before it is posted, so that swaps made from it no longer match the person, as judged by a held-out face recogniser. We'll explain the idea, show the results on 150 faces and two real face-swap models, and run a live demo."
- **Visual:** Clean dark title slide. Optional small side-by-side of original vs cloaked NASA portrait (screenshot from the demo; the difference is subtle, which is the point).
- **Layout:** Title centred, names below, URL bottom-right.

### Slide 2. The deepfake threat

- **On-slide:**
  - One photo is enough for a face swap (roop, FaceFusion, Deep-Live-Cam)
  - Every profile picture is raw material
  - Detection is reactive: the fake is already out
  - Our goal: a **proactive** defence applied before posting
- **Speaker notes (60–75 s):** "Modern face-swap tools are one-shot: give them one photo of you and any target video or image, and they produce a swap in seconds. Popular open-source tools like roop, FaceFusion and Deep-Live-Cam all use the same InsightFace model, inswapper. That means every photo you post is potential raw material. Most forensics work is about detection, which happens after the fake already exists and has spread. We look at the opposite side: can we change the photo before it is posted, so that it is useless as a deepfake source?"
- **Visual:** Simple icon flow: "your photo → public post → attacker downloads → face swap → fake". No real faces needed.
- **Layout:** Bullets left, icon flow right.

### Slide 3. How face swap works

- **On-slide:**
  - Swapper turns the source face into an **identity embedding** (512 numbers)
  - Generator paints that identity onto the target face
  - inswapper's identity = ArcFace **w600k_r50** embedding (public model)
  - Change the embedding → change the identity in the swap
- **Speaker notes (75–90 s):** "This is the key insight. A face swapper does not copy pixels from your photo. It first runs a face-recognition network on your face and gets an identity embedding, a vector of 512 numbers. The generator then paints that identity onto the target's face, keeping the target's pose, lighting, hair and skin. For inswapper, that recognition network is a public ArcFace model called w600k_r50; the swapper sees your identity only through it. So if we can make w600k_r50 produce the wrong embedding for your photo, the swapper paints the wrong identity. And because w600k_r50 is public, we can compute exact gradients through it: for inswapper, this is a white-box attack on its identity channel."
- **Visual:** Diagram: Source photo → [Face detector + alignment] → [ArcFace w600k_r50] → embedding vector → [Generator] ← Target image → Swapped face. Highlight the ArcFace box in violet as "the only door for identity".
- **Layout:** Full-width diagram, 2–3 bullets underneath.

### Slide 4. Our idea: the cloak

- **On-slide:**
  - Add a tiny perturbation: at most 8/255 per pixel (ε = 8)
  - Optimised so the embedding points **away** from the real you
  - Barely visible to people (LPIPS 0.020), very different to the recogniser
  - Swaps no longer match you, as judged by a held-out recogniser
- **Speaker notes (60–75 s):** "Our cloak is an adversarial perturbation. Each pixel value can change by at most 8 out of 255, which is subtle. But the change is not random: it is optimised so that the recogniser's embedding of the cloaked photo points away from the person's true embedding. The cosine similarity between them actually becomes negative. To a human the photo looks very similar, with low perceptual distortion (LPIPS 0.020 for our recommended setting), though the change may be visible on close inspection; to the swapper it is a different person. Random noise of the same strength does nothing: it gives 1.3 % protection, exactly the same as no cloak. The direction matters, not the amount."
- **Visual:** Three panels: original | cloaked | perturbation amplified ×10 (screenshot of the demo's "Can you see the difference?" section with a NASA example).
- **Layout:** Three images in a row, bullets below.

### Slide 5. Threat model

- **On-slide:**
  - Defender cloaks the **source** photo before posting
  - Attacker re-detects and re-aligns the face, may JPEG / resize / blur it
  - inswapper: encoder known (white-box) · SimSwap: unknown (black-box)
  - Judge: a **held-out** recogniser never used by the cloak
  - Out of scope: adaptive attackers, diffusion purification, target-side protection
- **Speaker notes (75–90 s):** "Here is exactly what we assume. The defender is the person posting the photo. The attacker downloads it, detects the face again, finds the landmarks again, and may compress, resize or blur the image before swapping. We test two swappers. For inswapper we know its identity encoder exactly, so that is the white-box case. SimSwap uses its own ArcFace network that we never touch, so it tests whether the cloak transfers to an unknown model, the black-box case. To decide whether a swap is protected, we use a third recogniser, glintr100, that the cloak never saw, and we compare the swap with a different photo of the same person. Out of scope: an attacker who knows our method and adapts to it, and diffusion-based purification. We return to these in the limitations."
- **Visual:** Threat-model diagram (spec in Section 4.2).
- **Layout:** Diagram top two-thirds, bullets bottom.

### Slide 6. Method pipeline

- **On-slide:**
  - Detect face + 5 landmarks (SCRFD)
  - Differentiable alignment to 112×112 (ArcFace) / 160×160 (FaceNet)
  - Ensemble: ArcFace w600k_r50 + FaceNet
  - PGD, 100 steps, with EOT + LPIPS → save as 8-bit PNG
- **Speaker notes (75–90 s):** "The pipeline: we detect the face and its five landmarks with SCRFD. The recognisers only see an aligned crop, so we wrote the alignment as a differentiable warp in PyTorch; gradients flow from the crop back to every pixel of the full photo. w600k_r50 is distributed as an ONNX file, which PyTorch cannot differentiate, so we converted it with onnx2torch and checked in a parity test that the converted model gives the same embeddings, cosine > 0.9999. We attack an ensemble of two recognisers, ArcFace and FaceNet, with projected gradient descent for 100 steps. Finally the image is rounded to 8-bit and saved as PNG, and we check that no pixel changed by more than 8."
- **Visual:** Pipeline diagram (spec in Section 4.1).
- **Layout:** Full-width horizontal pipeline.

### Slide 7. PGD in plain words

- **On-slide:**
  - Loss = cosine(embedding of cloaked photo, embedding of real you)
  - Step: nudge every pixel by ±0.8/255 in the direction that lowers the loss
  - Clip: never more than ±8/255 from the original
  - Repeat 100 times; keep the best result
- **Speaker notes (60–90 s):** "PGD, projected gradient descent, is simple. The loss is the cosine similarity between the embedding of the cloaked photo and the embedding of the real person; we want it as low as possible. At each step we compute the gradient and move every pixel by a small fixed amount, 0.8 out of 255, in the direction that lowers the loss; that's the 'sign of the gradient'. Then we project back: no pixel may differ from the original by more than 8 out of 255. We start from random noise inside that box and repeat 100 times. Every 10 steps we score the current result and keep the best one. We also tried two variants that did not help: pulling the embedding towards a decoy identity, like Fawkes does, and a term that tries to hide the face from the detector."
- **Visual:** Formula (large): $\delta \leftarrow \mathrm{clip}_{[-\epsilon,\epsilon]}\big(\delta - \alpha \cdot \mathrm{sign}(\nabla_\delta \mathcal{L})\big)$ with ε = 8/255, α = 0.8/255. Optional cartoon: a ball rolling downhill inside a square box.
- **Layout:** Formula centred, 4 bullets beneath.

### Slide 8. EOT + LPIPS

- **On-slide:**
  - EOT: at every step, randomly resize, blur and JPEG the photo, and jitter the landmarks
  - The cloak learns to survive what attackers (and social media) do
  - LPIPS term: penalise changes that a perceptual metric notices
  - Same protection, LPIPS 0.094 → 0.020
- **Speaker notes (75–90 s):** "Plain PGD is fragile: JPEG compression or a little blur wipes it out. EOT, Expectation over Transformation, fixes this. At every step we randomly transform the image before computing the loss: with probability one half we resize it down to between 50 and 100 % and back, with probability one half we blur it with sigma 0.5 to 1.5, and with probability one half we JPEG-compress it at quality 50 to 95. We also jitter the landmarks by about 2 pixels and 3 degrees, because the attacker re-detects them. So the cloak is optimised to work after these operations. The second addition is an LPIPS penalty. LPIPS is a learned measure of how different two images look to people. Adding it keeps protection the same while LPIPS falls from 0.094 to 0.020."
- **Visual:** Row of four small tiles: "resize 0.5–1.0", "blur σ 0.5–1.5", "JPEG 50–95", "landmarks ±2 px, ±3°", each with p = 0.5 (landmarks always). Below: a two-bar mini chart LPIPS 0.094 vs 0.020.
- **Layout:** Tiles top, bullets + mini chart bottom.

### Slide 9. Experimental setup

- **On-slide:**
  - 150 LFW identities × 3 targets = 450 swaps per setting
  - 2 swappers × 8 attacks × 10 post-processing transforms
  - Judge glintr100 (held out), threshold τ = 0.1564 at 1 % false-accept rate
  - 86,400 swap rows · Kaggle T4 · 2 sessions × ~7 h · commit e32b9b6
- **Speaker notes (75–90 s):** "We used Labeled Faces in the Wild. 150 source people each have two photos: photo A is cloaked, photo B is a different photo of the same person, used only as the reference. Each source is swapped onto three of 30 target faces, so 450 swaps per setting. We compare 8 attacks, including no cloak and random noise as controls, on two swappers, under 10 post-processing conditions. A swap counts as protected if glintr100, a recogniser the cloak never saw, scores it below a threshold set so that only 1 % of different-person pairs are accepted. That threshold is 0.1564, calibrated on LFW, where glintr100 is 99.84 % accurate. Confidence intervals bootstrap over people, not swaps. In total 86,400 swaps, run on free Kaggle T4 GPUs in two sessions of about 7 hours, all from one pinned commit."
- **Visual:** Small calibration table (glintr100 τ 0.1564, TAR 99.74 % at FAR 1 %, 10-fold accuracy 99.84 %) plus a "150 × 3 × 2 × 8 × 10" counter graphic.
- **Layout:** Two columns: setup bullets left, numbers right.

### Slide 10. Main results

- **On-slide:**
  - No cloak: 1.3 % protected · random noise: 1.3 %
  - inswapper (white-box encoder): **99.1–99.8 %** for all PGD variants except decoy (96.7 %)
  - SimSwap (black-box): 14.2 % → 26.0 % → **85.1 %** with EOT
  - Quality: PSNR 33.61 dB, LPIPS 0.020; 8.4 s per photo on a T4
- **Speaker notes (75–90 s):** "This is the main result. Grey bars on the left: without a cloak, only 1.3 % of swaps fail to match the person, so both swappers work well. Random noise of the same size: also 1.3 %. For inswapper, the white-box case, every PGD variant protects between 96.7 and 99.8 % of swaps, as judged by the held-out recogniser; the solid bars are all near the top. SimSwap, the faded bars, is the honest test. Attacking only ArcFace gives 14.2 %; adding FaceNet gives 26 %; adding EOT jumps to 85.1 %. With the LPIPS term we keep 85.1 % at LPIPS 0.020 and PSNR 33.61 dB. Our recommended setting takes 8.4 seconds per photo on a T4 GPU."
- **Visual:** `../web/public/results/main_psr.png` (solid = inswapper, faded = SimSwap). Annotate the EOT bars.
- **Layout:** Chart left two-thirds, 3–4 big numbers right.

### Slide 11. Robustness

- **On-slide:**
  - Without EOT: JPEG 50 → 50.2 %, blur σ=2 → 1.1 % (pgd_r50, inswapper)
  - With EOT: ≥ 98.7 % under all JPEG, resize, blur σ=1, median
  - Weak spots (pgd_ens_eot): blur σ=2 (72.2 %), bilateral filter (88.0 %)
  - Controls stay at 0.9–2.2 %: transforms alone don't "protect"
- **Speaker notes (75–90 s):** "Now we let the attacker post-process the cloaked photo. Each cell is the PSR after one transform. The plain single-model attack collapses: JPEG quality 50 halves it to 50.2 %, and a Gaussian blur with sigma 2 removes it completely, 1.1 %. With EOT, the rows stay green: at least 98.7 % under every JPEG quality, both resizes, mild blur and the median filter. For the EOT ensemble (pgd_ens_eot), the remaining weak spots are strong blur, 72.2 %, and the bilateral filter, 88.0 %, both stronger than what we trained against. The no-cloak and noise rows stay around 1 to 2 %, so the transforms by themselves don't make swaps fail."
- **Visual:** `../web/public/results/robustness_inswapper.png`. Highlight the pgd_r50 row and the pgd_ens_eot row.
- **Layout:** Chart full width, 2 callout boxes.

### Slide 12. Black-box transfer

- **On-slide:**
  - SimSwap uses its own ArcFace: we never touched it
  - With EOT: 85.1 % none, 81.6–85.1 % under JPEG/resize
  - Strong blur / bilateral (EOT variants): 46.4–48.7 %
  - Judges agree: glintr100 85.1 %, w600k_mbf 85.3 % (not independent), w600k_r50 86.7 % (white-box)
- **Speaker notes (75–90 s):** "This is the black-box case, SimSwap. Its identity network is different from anything we attacked, so success here means the cloak transfers. With EOT and LPIPS it protects 85.1 % of swaps, and keeps 81.6 to 85.1 % under JPEG and resizing. Strong blur and the bilateral filter bring it down to 46.4 to 48.7 %, which is our main limitation. The distribution plot shows why: without a cloak, similarity averages 0.460, far above the threshold line; with our cloak the bulk moves below it to about 0.045, with a tail still above. A second judge, w600k_mbf, gives 85.3 %, and the white-box judge 86.7 %; w600k_mbf shares training data with our attack model, so it is supporting rather than independent evidence. Two caveats: our SimSwap pipeline uses the same SCRFD landmarks as inswapper and a re-implemented paste-back without SimSwap's optional BiSeNet mask, and SimSwap's encoder is also an ArcFace-family network, so this is transfer across ArcFace-family encoders. And 14.9 % of SimSwap swaps stay unprotected."
- **Visual:** `../web/public/results/robustness_simswap.png` (main) and/or `../web/public/results/id_sim_distributions_simswap.png` (violins with τ line).
- **Layout:** Two charts side by side, or one chart with the violin plot as an inset.

### Slide 13. Trade-offs: ε and steps

- **On-slide:**
  - ε = 2: useless (1.6 %) · ε = 4: 47.3 % · ε = 8: 99.8 % (inswapper)
  - Bigger ε transfers better (SimSwap 64.9 % at ε = 16) but LPIPS 0.271
  - EOT + LPIPS at ε = 8 beats all of it: 85.1 % at LPIPS 0.020
  - More steps: weak, non-significant drop in transfer, SimSwap 33.1 → 29.6 → 26.0 % (20/50/100 steps)
- **Speaker notes (75–90 s):** "Two ablations on the ensemble attack without EOT. Budget: at 2 out of 255 nothing happens; at 4 we reach 47.3 % on inswapper but it doesn't survive JPEG; at 8 we saturate at 99.8 %. Increasing epsilon helps SimSwap, up to 64.9 % at 16, but the image gets visibly worse, LPIPS 0.271. EOT with LPIPS at epsilon 8 is better on both axes. Steps: more steps make the attack stronger against our own models, but SimSwap protection goes down, from 33.1 to 29.6 to 26.0 %. This is a weak trend: the confidence intervals overlap, so it is not significant, but it is consistent with the attack overfitting to the models it can see. It suggests the goal is not simply to minimise our own loss as hard as possible, but to find changes that generalise."
- **Visual:** `../web/public/results/ablation_psr_vs_lpips_simswap.png` (left) and `../web/public/results/ablation_steps_simswap.png` (right). Alternatives: `ablation_eps_inswapper.png`, `ablation_eps_simswap.png`, `ablation_steps_inswapper.png`.
- **Layout:** Two charts side by side, bullets as captions.

### Slide 14. Live demo

- **On-slide:**
  - face-cloak.vercel.app → Live demo
  - Example victim: Ellen Ochoa or Raja Chari (public-domain NASA portraits)
  - Watch: verdict, slider, two deepfakes, identity bars vs τ
  - White-box best case (inswapper); SimSwap numbers on Results page
- **Speaker notes (60–90 s, plus 2–3 min demo):** "Let's see it live. I pick a public-domain NASA portrait, Ellen Ochoa, tick consent, and click Cloak and attack. The backend detects the face, cloaks it, swaps both the original and the cloaked photo onto another astronaut, and judges both with glintr100. Left: the deepfake from the original; it clearly carries her identity, score 0.753 in our recorded run. Right: the deepfake from the cloaked photo; score −0.388, below the threshold line, so the verdict is Protected. The slider shows the two inputs are very close; the change is subtle, though it may be visible on close inspection. One honest note: the demo uses inswapper, which is our white-box best case; the harder SimSwap numbers are on the Results page." (Full runbook in Section 5.)
- **Visual:** Live browser, or a pre-recorded screenshot of the result panel (verdict + two deepfakes + identity bars) as a fallback slide.
- **Layout:** Single large screenshot; switch to the browser.

### Slide 15. Limitations and ethics

- **On-slide:**
  - Strong blur / bilateral filter → SimSwap 46.4–48.7 % (EOT variants)
  - Not tested: adaptive attackers, diffusion purification (DiffPure)
  - LFW only (low-res celebrities) · one seed · source path only
  - Ethics: research-only weights not redistributed, no faces in repo, demo targets fixed public-domain, watermark, rate limits, 10-min in-memory
- **Speaker notes (75–90 s):** "What this does not solve. A determined attacker can blur the photo strongly or use a bilateral filter, and SimSwap protection drops to about half (46.4–48.7 %). We did not test an attacker who knows our method, or a diffusion model that purifies the image first; those are the most likely ways around it, so our numbers are an upper bound. LFW photos are small celebrity images, we ran one random seed, and we only protect the source photo. On ethics: the model weights are non-commercial research licences, so we never redistribute them; no face data is in the repository; the demo can only swap onto fixed public-domain NASA portraits, watermarks every output as synthetic, rate-limits users and keeps uploads in memory for at most ten minutes."
- **Visual:** Two columns with icons: "Limitations" and "Safeguards".
- **Layout:** Two-column text slide.

### Slide 16. Conclusion

- **On-slide:**
  - Face swappers inherit the weaknesses of their recognisers
  - inswapper: 99.3 % protected · SimSwap: 85.1 % (black-box)
  - EOT is the key ingredient; LPIPS makes it harder to see
  - Next: purification, adaptive attacks, more swappers, human study
- **Speaker notes (60 s):** "To conclude: one-shot face swappers depend on a face-recognition embedding, and that dependency is a weakness we can exploit defensively. A cloak of 8 out of 255 protects 99.3 % of inswapper swaps and transfers to an unseen swapper in 85.1 % of cases, with low perceptual distortion. EOT is what makes it both robust and transferable. Future work: test against diffusion purification and adaptive attackers, more swappers, and a human perception study."
- **Visual:** Three large number tiles: 1.3 % → 99.3 % (inswapper), 85.1 % (SimSwap), LPIPS 0.020.
- **Layout:** Number tiles centred, bullets below.

### Slide 17. Team

- **On-slide:**
  - Jacob Antony Jeejo · Manasa Manoj · Praneetha Reddy
  - Equal contribution
  - Indian Institute of Information Technology Kottayam, Multimedia Forensics
  - Code: github.com/J4KE-B/face-cloak · Demo: face-cloak.vercel.app
- **Speaker notes (15–20 s):** "This was a three-person project with equal contribution. The code, report and demo are all public at these links."
- **Visual:** Three name cards (no photos needed), links, optional QR code to the live site.
- **Layout:** Three equal cards in a row.

### Slide 18. Q&A

- **On-slide:**
  - Questions?
  - face-cloak.vercel.app · github.com/J4KE-B/face-cloak
- **Speaker notes:** Use the prepared answers in Section 6. Keep backup slides with Table 1 (main results) and the full robustness matrix ready.
- **Visual:** `../web/public/results/id_sim_distributions_inswapper.png` faded in the background, or plain.
- **Layout:** Large "Questions?" centred.

---

## 3. Key-numbers cheat sheet

| What | Number |
|---|---|
| Faces / swaps per setting | 150 LFW identities × 3 targets = 450 |
| Total swap rows | 86,400 (72,000 main + 9,000 ε ablation + 5,400 steps ablation) |
| Compute | Kaggle T4, 2 sessions × ~7.05 h, commit e32b9b6 |
| Budget / step | ε = 8/255, α = ε/10 = 0.8/255, 100 steps, seed 0 |
| Judge threshold | glintr100 τ = 0.1564 at FAR 1 % (TAR 99.74 %, 10-fold acc. 99.84 %) |
| No cloak / random noise PSR | 1.3 % / 1.3 % (both swappers) |
| inswapper PSR, recommended (EOT + LPIPS) | 99.3 % [98.0, 100.0] |
| inswapper PSR, best | 99.8 % (pgd_ens) |
| SimSwap PSR: r50 only → ensemble → + EOT | 14.2 % → 26.0 % → 85.1 % |
| SimSwap PSR, recommended | 85.1 % [79.6, 90.0] |
| Quality, recommended | PSNR 33.61 dB, SSIM 0.851, LPIPS 0.020 |
| LPIPS without / with LPIPS term | 0.094 / 0.020 |
| JPEG 50, inswapper: plain r50 / EOT | 50.2 % / 98.7 % |
| Blur σ = 2, inswapper: plain r50 / EOT + LPIPS | 1.1 % / 71.8 % |
| SimSwap EOT + LPIPS under blur σ = 2 / bilateral | 48.7 % / 46.4 % |
| Steps 20 / 50 / 100, SimSwap | 33.1 % / 29.6 % / 26.0 % (weak, non-significant trend) |
| ε 2 / 4 / 8, inswapper | 1.6 % / 47.3 % / 99.8 % |
| ε 16, SimSwap (no EOT) | 64.9 % at LPIPS 0.271 |
| Decoy (targeted) variant | 96.7 % inswapper (lower than 99.8 %), 22.7 % SimSwap (no detectable difference from 26.0 %) |
| Detector term | face still detected 100 %; SCRFD score 0.834 → 0.809; PSR unchanged |
| Cloak time per photo (T4) | 1.6 s (r50) · 5.2 s (ensemble) · 8.4 s (recommended) |
| Demo examples (identity score, uncloaked → cloaked) | Ochoa 0.753 → −0.388 · Chari 0.851 → −0.372 · Menon 0.841 → −0.226 · Chawla 0.845 → 0.006 |
| Demo CPU run | about 55 s |

---

## 4. Diagram specs

Colours: violet/indigo for "our method" boxes, cyan for "attacker" boxes, grey for neutral data, red outline for "held-out judge".

### 4.1 Method pipeline (slide 6)

Left-to-right, one row of boxes, arrows between consecutive boxes:

1. **[Grey] Clean photo x₀** (small face thumbnail).
2. **[Violet] SCRFD detector** → label under arrow: "5 landmarks".
3. **[Violet] Differentiable alignment** (similarity transform → grid_sample) → two arrows split:
   - upper: "112×112" → **[Violet] ArcFace w600k_r50** (note: "= inswapper's identity encoder; ONNX → PyTorch")
   - lower: "160×160" → **[Violet] FaceNet (VGGFace2)**
4. Both recogniser boxes → **[Violet] Loss** box: "cos(emb, clean emb) + LPIPS".
5. Loss → **[Violet] PGD update** box: "δ ← clip(δ − 0.8/255 · sign(∇)) , |δ| ≤ 8/255".
6. A curved arrow from PGD update back to step 2 labelled "×100 steps".
7. Above the loop, a dashed box **[Cyan outline] EOT** with four chips: "resize 0.5–1.0", "blur σ 0.5–1.5", "JPEG 50–95", "landmark jitter 2 px / 3°", with an arrow pointing into the alignment box ("random transform each step").
8. Exit arrow from PGD (labelled "best of every 10 steps") → **[Grey] Round to 8-bit PNG** → **[Grey] Cloaked photo x̃** (thumbnail looks very close to x₀).

### 4.2 Threat model (slide 5)

Three lanes, left to right.

**Lane 1, Defender (violet):**
- Box "Photo A of the person" → arrow → Box "face-cloak (PGD + EOT + LPIPS)" → arrow → Box "Cloaked photo Ã (posted online)".

**Lane 2, Attacker (cyan):** arrow from "Cloaked photo Ã" into:
- Box "Optional post-processing: JPEG 90/75/50, resize 0.75/0.5, blur σ 1/2, median, bilateral" → arrow →
- Box "Re-detect face + landmarks, re-align" → arrow →
- Box "Face swapper" with two labelled sub-boxes: "inswapper_128 (encoder = w600k_r50: WHITE-BOX)" and "SimSwap (own ArcFace: BLACK-BOX)".
- A separate box "Target image T" with an arrow into the swapper.
- Output arrow → Box "Swapped face".

**Lane 3, Evaluation (red outline):**
- Box "Photo B of the same person (never cloaked)".
- Both "Swapped face" and "Photo B" → arrows → Box "Held-out judge glintr100 (never used by the cloak)" → arrow → Diamond "cosine < τ = 0.1564?" → "Yes: protected" / "No: not protected".

Footnote under the diagram: "Out of scope: adaptive attacker, diffusion purification, protecting the target image."

---

## 5. Live-demo runbook

**Start the backend 10–15 minutes before presenting.** The website (Home, Results, About) always works; only the Live demo needs a backend.

### Option A: Kaggle GPU (recommended, fast)

1. Open https://face-cloak.vercel.app → **Live demo** → click **Open the demo notebook on Kaggle**. Kaggle copies it as a private notebook.
2. Right-hand panel: **Accelerator: GPU T4**, **Internet: On**, **Add-ons → Secrets:** add `DEMO_KEY` (your passcode, at least 12 characters).
3. Click **Run all**. After about 3–5 minutes the last cells print a tunnel URL and an **auto-connect link**.
4. Open the auto-connect link, type the passcode, click **Connect**. The badge should show **Online** and `GPU (cuda)`.
5. Keep the Kaggle tab open. Never make the notebook public or save a version with outputs (the outputs contain the tunnel URL).

### Option B: laptop (slower, CPU)

1. From the repo folder run: `PORT=8765 ./server/run_local.sh` (any free port works; the script's default is 8000).
2. It prints a tunnel URL, a passcode and an auto-connect link. Open the link, enter the passcode, **Connect**.
3. Expect about 55 s per run on CPU (fewer steps by default). Press Ctrl-C to stop.

### Running it

1. Click the one-click example victim **Ellen Ochoa** (recommended target: Kim) or **Raja Chari** (target: Jemison).
2. Tick the **consent** box.
3. Leave defaults (ε = 8; mode Fast or Robust; Robust = r50 + FaceNet + EOT).
4. Click **Cloak and attack** and narrate the progress bar: detecting → cloaking → swapping → judging.

### What to point at

- **Verdict**: "Protected".
- **Slider** ("Can you see the difference?"): original vs cloaked look very close (the change is subtle but may be visible on close inspection); the side panel shows the change amplified ×10.
- **Two deepfakes**: left from the original (looks like the victim), right from the cloaked photo (does not). Both are labelled SYNTHETIC.
- **Identity bars vs τ**: original swap well above the white τ line, cloaked swap below it. Recorded values: Ochoa 0.753 → −0.388; Chari 0.851 → −0.372 (a live run gave 0.848 → −0.287).
- **Image quality**: PSNR, SSIM, LPIPS and L∞ ≤ ε. Our recommended setting gives PSNR 33.61 dB with LPIPS 0.020: subtle, but it may be visible on close inspection. Do not claim it is invisible.

**Say explicitly:** "This is inswapper, the white-box best case. The black-box SimSwap numbers (85.1 %) are on the Results page."

### Fallbacks

- Backend down: type `mock` as Backend URL (any passcode). It shows the interface with fake numbers and labels the result **MOCK DATA**. Say so; never present it as a real result.
- Or switch to the **/results** page and walk through the charts.
- Keep a screenshot of a real result panel in the deck (slide 14) as a last resort.

---

## 6. Anticipated viva questions

**Q1. Why is there no training? Where is the model?**
The cloak is an optimisation per photo, not a trained network. For each photo we run 100 steps of PGD against fixed, pretrained recognisers. Nothing is learned across photos; 8.4 s on a T4 per photo for the recommended setting.

**Q2. What do white-box and black-box mean here?**
White-box: we have the exact identity encoder the swapper uses. inswapper's identity input is the w600k_r50 ArcFace embedding, and we attack w600k_r50 directly (we do not need the generator). Black-box: SimSwap uses its own ArcFace network that we never used; any protection there is transfer. inswapper 99.3 %, SimSwap 85.1 %.

**Q3. Why a held-out judge?**
If we judged with the model we attacked, we would only show that we fooled our own model. glintr100 is never used in the attack (the code refuses it), so a low score means the identity was removed as far as an unseen recogniser can tell. On inswapper the white-box judge w600k_r50 is equal or higher than glintr100 for every PGD attack, the expected ordering.

**Q4. What is τ and "FAR 1 %"?**
τ is the similarity threshold above which the judge says "same person". We set it on LFW so that only 1 % of different-person pairs pass (false-accept rate 1 %). For glintr100, τ = 0.1564; at that threshold it accepts 99.74 % of same-person pairs. A swap below τ is not recognised as the person.

**Q5. Why is SimSwap protection lower?**
Its identity network is different, so the perturbation must transfer. Without EOT it transfers poorly (14.2 % single model, 26.0 % ensemble); with EOT it reaches 85.1 %. A plausible but untested explanation is that EOT suppresses fine, model-specific perturbation detail. Caveat: EOT runs also pick the best iterate by loss over 8 fixed EOT samples (non-EOT runs use the clean loss), so the jump reflects EOT gradients plus EOT-based selection. SimSwap's encoder is also ArcFace-family, and our SimSwap pipeline shares the SCRFD landmarks with inswapper, so this is transfer across ArcFace-family encoders.

**Q6. Does it survive JPEG / social media?**
With EOT, yes for JPEG and resizing: inswapper ≥ 98.7 % even at JPEG quality 50; SimSwap 81.6–85.1 %. Without EOT, JPEG 50 halves it (50.2 %). We did not test specific platforms' pipelines, only these standard operations. Save the cloaked image as PNG where possible.

**Q7. Can the attacker purify or denoise it?**
Partly, yes. Strong blur (σ = 2) and bilateral filtering cut SimSwap protection to 46.4–48.7 % (inswapper 71.8–88.0 % across the two EOT variants). We did not test diffusion purification (e.g., DiffPure) or an adaptive attacker who knows our method; those are likely to weaken it further. It raises the cost; it is not a guarantee.

**Q8. Why LPIPS and not just PSNR?**
PSNR counts pixel error equally everywhere; LPIPS measures how different images look to a learned perceptual model, which correlates better with human judgement. With the LPIPS term, PSNR barely changes (33.59 → 33.61 dB) but LPIPS drops 0.094 → 0.020 at the same PSR. Caveat: LPIPS is also what we optimise, and we did not run a human study.

**Q9. Why the NASA examples? Why did a selfie look unconvincing?**
inswapper works at 128 px and transfers only inner-face identity; skin tone, hair, face shape and lighting come from the target. On a low-light, tilted selfie the judge said "protected" (uncloaked swap well above τ, cloaked about 0), but even the uncloaked deepfake looked mostly like the target to people, so the demo was not convincing. The public-domain NASA portraits give clear, convincing uncloaked swaps, so the effect of the cloak is visible.

**Q10. Couldn't this be misused?**
The cloak removes identity; it cannot create a deepfake. The risk of the demo is the swapper, so it only swaps onto fixed public-domain portraits, watermarks outputs "SYNTHETIC", requires consent and a passcode, rate-limits (5 jobs/h per IP, 40/h total) and keeps uploads in memory for at most 10 minutes. Research-only weights are never redistributed.

**Q11. How does this compare with Fawkes?**
Fawkes protects against face-recognition models (pushing photos towards a decoy identity). We protect against face swapping, where the recogniser is inside the swapper. We tried a Fawkes-style decoy term: PSR was lower on inswapper (96.7 % vs 99.8 %, non-overlapping CIs), with no detectable difference on SimSwap (22.7 % vs 26.0 %, CIs overlap). We did not run Fawkes itself, so we make no direct comparison.

**Q12. What did the compute cost?**
Free Kaggle T4 GPUs: two sessions of about 7.05 h each for 86,400 swaps; calibration about 25 min. Per photo: 1.6 s (single model) to 8.4 s (recommended).

**Q13. Why didn't the detector term work?**
The idea was to make the attacker's detector miss the face. With an 8/255 budget it only lowered the mean SCRFD score from 0.834 to 0.809; the face was still detected in 100 % of images, PSR did not improve, and LPIPS got worse (0.170). A clear face is far above the detection threshold.

**Q14. Why do more PGD steps hurt transfer?**
More steps push our own models harder (cosine to clean −0.286 → −0.585), but the perturbation specialises to their quirks, so it may transfer less: SimSwap 33.1 → 29.6 → 26.0 % for 20/50/100 steps. This is a weak, non-significant downward trend (20 vs 100 steps: 33.1 [26.9, 39.3] vs 26.0 [20.0, 31.8]); the 'none' and JPEG 75 results use the same cloaks, so they are not independent replications. It is consistent with overfitting to the attack models, not proof of it.

**Q15. Is the 8/255 limit really respected?**
Yes. The output is rounded to 8-bit, saved as PNG and checked: the maximum pixel change is exactly 8 for every cloak (and exactly 2, 4, 12, 16 in the ε ablation).

---

## 7. Design guidance

- **Theme:** dark, matching the website: near-black background (e.g., #0B0B12), off-white text, accents in violet/indigo (e.g., #7C5CFF / #6366F1) and cyan (e.g., #22D3EE). Use violet for "our method", cyan for "attacker", red only for failure/limitations.
- **One idea per slide.** At most 4–5 bullets, each under about 10 words. The speaker notes carry the detail.
- **Large numbers.** Put the headline number of each results slide in 60–80 pt (e.g., "85.1 %").
- **Cite charts.** Every chart gets a small caption: "PSR, glintr100 judge, n = 450, 95 % CI". The chart PNGs have a white background: place them on a white rounded card, or crop tightly.
- **Labels.** Always write "inswapper (white-box encoder)" and "SimSwap (black-box)" on result slides.
- **Faces.** Use only the public-domain NASA demo portraits or screenshots from the demo. Never show LFW faces or swaps of them.
- **Fonts:** one sans-serif family (e.g., Inter); formulas only on slide 7.

## 8. Assets list

| Asset | Path / link | Used on |
|---|---|---|
| Main PSR chart | `../web/public/results/main_psr.png` | 10 |
| Robustness, inswapper | `../web/public/results/robustness_inswapper.png` | 11 |
| Robustness, SimSwap | `../web/public/results/robustness_simswap.png` | 12 |
| id-sim distributions, inswapper | `../web/public/results/id_sim_distributions_inswapper.png` | 18 / backup |
| id-sim distributions, SimSwap | `../web/public/results/id_sim_distributions_simswap.png` | 12 |
| ε ablation, inswapper / SimSwap | `../web/public/results/ablation_eps_inswapper.png`, `../web/public/results/ablation_eps_simswap.png` | 13 / backup |
| PSR vs LPIPS, inswapper / SimSwap | `../web/public/results/ablation_psr_vs_lpips_inswapper.png`, `../web/public/results/ablation_psr_vs_lpips_simswap.png` | 13 |
| Steps ablation, inswapper / SimSwap | `../web/public/results/ablation_steps_inswapper.png`, `../web/public/results/ablation_steps_simswap.png` | 13 |
| Live site | https://face-cloak.vercel.app | 1, 14, 17 |
| Repository | https://github.com/J4KE-B/face-cloak | 17, 18 |
| Report | `docs/REPORT.md` | backup / handout |
| Website guide | `docs/WEBSITE_GUIDE.md` | demo preparation |
| Demo screenshots | take from the Live demo with Ellen Ochoa / Raja Chari | 1, 4, 14 |
