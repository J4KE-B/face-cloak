# How to use the face-cloak website

Website: **https://face-cloak.vercel.app**

The site has four pages:

| Page | What it shows |
|---|---|
| **Home** | What the project does, in plain words. |
| **Live demo** | Upload a photo, cloak it, and watch a face-swap attack succeed on the original but fail on the cloaked copy. |
| **Results** | Numbers and figures from the full experiment on 150 LFW faces. |
| **About** | Method, limitations, ethics, team and credits. |

**Home, Results and About** always work. **Live demo** needs a *backend* running first. The face-swap models are too big for the website itself, so they run on a Python server. Start the backend 5–10 minutes before you present.

---

## 1. Start the backend

### Option A: Kaggle GPU (recommended, about 5–15 s per photo)

1. On the **Live demo** page, click **Open the demo notebook on Kaggle**. Kaggle copies the notebook into *your* account as a **private** notebook.
2. In the notebook's right-hand panel:
   - **Accelerator:** GPU T4
   - **Internet:** On (Kaggle asks you to verify your phone number once)
   - **Add-ons → Secrets:** add a secret named `DEMO_KEY`. Its value is your **passcode**, at least 12 characters, for example `forensics-demo-2026`.
3. Click **Run all**. After about 3–5 minutes the last cells print:
   - a **tunnel URL** like `https://something-random.trycloudflare.com`
   - an **auto-connect link** like `https://face-cloak.vercel.app/demo?backend=...`
4. Keep the Kaggle tab open while you present. The backend only lives as long as the notebook runs.

> ⚠️ Never make this notebook public, and never use "Save Version" with outputs. The outputs contain your live tunnel URL.

### Option B: your laptop (slower, about 1–3 min per photo on CPU)

From the repository folder:

```bash
./server/run_local.sh
```

It prints a tunnel URL, a passcode and an auto-connect link. Press **Ctrl-C** to stop. Closing the terminal also stops the backend and the tunnel.

---

## 2. Connect the website

- **Easiest:** open the **auto-connect link** printed by the notebook or script. The backend URL is filled in for you. Type the **passcode** and click **Connect**.
- **Manually:** on **Live demo**, paste the tunnel URL into **Backend URL**, type the passcode and click **Connect**.

The badge turns **Online** and shows `GPU (cuda)` or `CPU` plus the queue length.

> 🔒 **Only connect to a backend you started yourself.** Anyone can create a `trycloudflare.com` address, so a link from someone else could send your photo to their server. The site shows which host your photo will go to before you upload. Check that it matches the URL your notebook printed.

**Want to show the interface without a backend?** Type `mock` as the Backend URL, with any passcode. Everything runs in your browser with fake numbers, and the result says "MOCK DATA". Use this only to show the interface, never as a real result.

---

## 3. Run the demo

1. **Your photo:** drag in a clear, roughly front-facing photo with one face. It is shrunk to 1024 px and its metadata (GPS, camera) is removed in your browser before upload.
2. **Second photo of you (optional, recommended):** a different photo of the same person. The judge compares the swaps against this photo. Without it, the judge compares against the uploaded photo itself, which makes the result look better than it really is.
3. **Consent:** tick the box. Only upload photos of yourself, or of people who agreed to it.
4. **Target portrait:** pick one of the fixed public-domain NASA astronaut portraits. Your face gets swapped onto it. You cannot upload your own targets, by design.
5. **Settings:**
   - **Strength ε:** the maximum change per pixel, out of 255. The default is 8. Higher values protect more but are more visible.
   - **Steps:** the number of optimisation rounds. More steps give a stronger cloak but take longer.
   - **Mode:** **Fast** attacks one face-recognition model. **Robust** adds random JPEG, blur and resize during optimisation, plus a second model. It is slower and survives compression better.
6. Click **Cloak and attack**. A progress bar steps through *detecting → cloaking → swapping → judging*.

---

## 4. Read the result

| Section | Meaning |
|---|---|
| **Verdict** | ✅ **Protected**: the deepfake made from the cloaked photo no longer matches your identity. ❌ **Not protected**: it still does. ⚠️ **Inconclusive**: even the deepfake from the original photo didn't match, so the comparison doesn't tell us anything. |
| **Can you see the difference?** | Drag the slider between the original and cloaked photos. The panel beside it shows the invisible change, amplified ×10. |
| **What the attacker gets** | Left: the deepfake from your original photo. Right: the deepfake from the cloaked photo. Both carry a **SYNTHETIC** label. |
| **Identity score** | How much each deepfake still looks like you, scored by a separate model (glintr100) that the cloak never saw. The white line is the threshold τ: **above τ = still you, below τ = someone else**. |
| **Image quality** | PSNR (higher means closer to the original; above 35 dB is hard to see), SSIM (1.0 means identical), LPIPS (lower is better) and L∞ (the largest change to any pixel, at most ε). |

Click **Download cloaked PNG** to save the protected photo. Keep it as PNG: JPEG compression weakens the cloak.

**Honest framing for the viva:** the live demo attacks with *inswapper*, whose identity model is the one the cloak was optimised against. That makes it the **white-box best case**. The **Results** page reports the harder **black-box** case (SimSwap) and the robustness tests.

---

## 5. Privacy

- Photos go only to the backend you connected, never to the website or Vercel.
- The backend keeps everything in memory, deletes each job as soon as the result is shown, and drops everything after 10 minutes at most. Nothing is written to disk.
- There are no analytics or trackers on the site.

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| **"Backend unreachable"** | The Kaggle notebook or laptop script has stopped, or the URL is wrong. Restart it, then use the new URL. Tunnel URLs change every run. |
| **"Wrong passcode" (401)** | Use exactly the `DEMO_KEY` you set. After about 20 wrong tries the backend blocks you for a few minutes (429). |
| **"Too many requests" (429)** | Each visitor can run 5 jobs per hour. Wait for the countdown. |
| **"Busy" (503)** | Another job is running. The backend handles one photo at a time. |
| **"No face found" (400)** | Use a clearer, front-facing photo with one face. |
| **"File too large / unsupported" (413/415)** | Use JPEG, PNG or WebP under 5 MB. iPhone HEIC photos are converted automatically when picked from the file dialog. |
| **Very slow** | You are on the CPU (laptop) backend. Use the Kaggle GPU notebook, or lower the steps. |
| **`http://127.0.0.1` backend doesn't work from the live site** | Browsers block or prompt for this (mixed content / local-network permission). Use the tunnel URL instead. |

---

More detail: [backend reference](../server/README.md) · [running the full experiments on Kaggle](../KAGGLE_RUN.md) · [frontend developer notes](../web/README.md)
