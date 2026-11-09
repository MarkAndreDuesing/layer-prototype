# Sketch-to-image on CPU with FastSD CPU: install, setup and the 3 paths

> **Archived (2026-11-09).** This is the first, machine-specific version of the guide. It still has Path 3 (ControlNet), our test prompt and the Hugging Face cache download, which does **not** work offline (`IncompleteSnapshotError`). Use the current two-path guides instead: [../fastsd-cpu/INSTALL.md](../guides/fastsd-cpu/INSTALL.md) and [../fastsd-cpu/USAGE.md](../guides/fastsd-cpu/USAGE.md). It is kept because it records the exact values behind the test runs.

Based on the test runs (see `experiments/CONFIRMATION_RESULTS.md` and `experiments/MODEL_RANKING_SHEET.xlsx`).
Tested on Windows 11, 15.7 GB RAM, no GPU. Values below are the ones used in the tests, not guesses.

| Path | What | Time per image | RAM | Disk (model) |
|---|---|---|---|---|
| 1 | SD-Turbo img2img | ~20 s | about half of Path 2 | 4.9 GB |
| 2 | SDXL-Turbo img2img (or text-only) | ~58 s | ~11 GB | 13 GB |
| 3 | SD 1.5 + LCM-LoRA + scribble ControlNet | not recorded | not recorded | 5.2 GB + 129 MB + 723 MB |

Free disk needed: about 25 GB for everything. Close Chrome, IntelliJ and virtual machines before Path 2.

---

# PART A: Install and setup (once)

## A1. Install the tools
1. Install Python 3.11.9: <https://www.python.org/downloads/release/python-3119/> (tick "Add python.exe to PATH").
2. Install Git: <https://git-scm.com/download/win>
3. Install uv, in a new PowerShell window:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
4. Open a new PowerShell window and check all three:
   ```powershell
   python --version
   git --version
   uv --version
   ```
   `python --version` must say 3.11.x.

## A2. Get the code
1. ```powershell
   cd "$HOME\Documents\StabeDiffusionCPU"
   git clone https://github.com/rupeshs/fastsdcpu.git
   cd fastsdcpu
   ```

## A3. Create the environment (do this instead of `install.bat`)
`install.bat` asks uv for Python 3.11.6 and failed here. These commands use your installed Python instead.

1. Create the environment from the installed Python:
   ```powershell
   uv venv --python 3.11 --no-managed-python env
   ```
2. Install the project requirements:
   ```powershell
   uv pip install --python env\Scripts\python.exe -r requirements.txt
   ```
3. Reinstall the pinned CPU build of torch. The requirements pull in a newer torch, which was not used in the tests:
   ```powershell
   uv pip install --python env\Scripts\python.exe torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu --reinstall-package torch --reinstall-package torchvision --no-deps
   ```
4. Check the key versions:
   ```powershell
   uv pip list --python env\Scripts\python.exe | findstr /i "torch numpy pillow diffusers transformers gradio"
   ```
   Expected: torch 2.8.0+cpu, torchvision 0.23.0+cpu, numpy 1.26.4, pillow 9.4.0, diffusers 0.37.0, transformers 5.0.0, gradio 6.18.0.

## A4. Remove scikit-learn (Windows Smart App Control)
Smart App Control blocked an unsigned scikit-learn file with the error "An Application Control policy has blocked this file". Only `nncf` wants scikit-learn, and the app runs without it. Do not change the Windows security setting.

1. ```powershell
   uv pip uninstall --python env\Scripts\python.exe scikit-learn
   ```
2. If a later error names another blocked file, uninstall the package that file belongs to the same way.

## A5. Apply the SDXL-Turbo fix (local patch)
Without it, SDXL-Turbo fails with "mat1 and mat2 must have the same dtype, but got Half and Float".

1. Open `fastsdcpu\src\backend\lcm_text_to_image.py` and find this line (about line 515):
   ```python
   if not self.use_openvino and not is_openvino_device():
       self._pipeline_to_device()
   ```
2. Add the `elif` straight after it, at the same indentation:
   ```python
   if not self.use_openvino and not is_openvino_device():
       self._pipeline_to_device()
   elif not self.use_openvino:
       # transformers>=5 keeps the stored (fp16) dtype for text encoders,
       # so cast every component explicitly to avoid Half/Float mismatches
       for component in self.pipeline.components.values():
           if isinstance(component, torch.nn.Module):
               component.to(dtype=self.torch_data_type)
   ```
3. A `git pull` overwrites this edit, so redo it after every update.

## A6. Download the models
The built-in downloader ran out of memory, so use the plain one with the `xet` downloader off. The first generation would download these anyway, but doing it here avoids a failed first run.

1. Download the models (PowerShell, one block):
   ```powershell
   $env:HF_HUB_DISABLE_XET = "1"
   env\Scripts\python.exe -c "from huggingface_hub import snapshot_download as d; [d(m, max_workers=2) for m in ['stabilityai/sd-turbo','stabilityai/sdxl-turbo','stable-diffusion-v1-5/stable-diffusion-v1-5','latent-consistency/lcm-lora-sdv1-5']]"
   ```
   Skip any model for a path you will not use. Sizes are in the table at the top.
2. Download the scribble ControlNet (723 MB, Path 3 only): <https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/blob/main/control_v11p_sd15_scribble_fp16.safetensors>
3. Save it as `fastsdcpu\controlnet_models\control_v11p_sd15_scribble_fp16.safetensors`.
4. The tests used `stable-diffusion-v1-5/stable-diffusion-v1-5` as the Path 3 base model. It is not in the app's dropdown, so add it as a new last line of `fastsdcpu\configs\stable-diffusion-models.txt`:
   ```
   stable-diffusion-v1-5/stable-diffusion-v1-5
   ```
   (the app reads this file when it starts).

## A7. Start the app
1. ```powershell
   cd "$HOME\Documents\StabeDiffusionCPU\fastsdcpu"
   .\start-webui.bat
   ```
2. Wait for "Running on local URL".
3. Open <http://127.0.0.1:7860>.
4. To stop it later, click the terminal and press `Ctrl+C`.

---

# PART B: Prepare your inputs (once per sketch)

## B1. Make the two input images
1. Save your hand-drawn sketch (plain) and the colour-filled version (night sky, moon, green trees) as `sketch_plain.png` and `sketch_filled.png` in a folder `inputs`.
2. Flatten each onto white and resize to 512×512:
   ```powershell
   cd "$HOME\Documents\StabeDiffusionCPU\fastsdcpu"
   env\Scripts\python.exe -c "from PIL import Image; import sys; [Image.alpha_composite(Image.new('RGBA', (i:=Image.open(f'../inputs/{n}.png').convert('RGBA')).size, 'white'), i).convert('RGB').resize((512,512)).save(f'../inputs/{n}_512.png') for n in ['sketch_plain','sketch_filled']]"
   ```
3. Use `*_512.png` below. Which one to upload:

| Input | Gives | Use for |
|---|---|---|
| `sketch_filled_512.png` | Night scene with moon, layout kept, style words have less effect | Paths 1 and 2 |
| `sketch_plain_512.png` | Pale daytime scene, strongest impressionist results | Path 2 variant, and Path 3 |

Do not blur the input. It only defocused the results.

## B2. Prompts (same scene text, only the style prefix changes)
Copy one of these into the prompt box:

1. No style:
   ```
   one skeleton dancing next to a large campfire with a stone ring, campfire in the center, skeleton on the right side, night forest of tall pine trees, full moon and stars
   ```
2. Pixel art:
   ```
   pixel art, retro 16-bit game scene, one skeleton dancing next to a large campfire with a stone ring, campfire in the center, skeleton on the right side, night forest of tall pine trees, full moon and stars
   ```
3. Impressionist:
   ```
   impressionist oil painting, visible brushstrokes, soft light, one skeleton dancing next to a large campfire with a stone ring, campfire in the center, skeleton on the right side, night forest of tall pine trees, full moon and stars
   ```
4. Hyperrealistic (you kept none of these, 0 of 30):
   ```
   hyperrealistic photograph, ultra detailed, cinematic lighting, one skeleton dancing next to a large campfire with a stone ring, campfire in the center, skeleton on the right side, night forest of tall pine trees, full moon and stars
   ```

Leave the negative prompt empty on Paths 1 and 2. Guidance 1 means the Turbo models ignore it.

---

# PART C: The three paths

Each guide starts from a fresh page at <http://127.0.0.1:7860>. Settings reset when the server restarts, so set every value.

## Path 1: SD-Turbo img2img (fast, weaker layout)
1. Under **Mode**, select **LCM**.
2. Open **Models** and set **LCM model** to `stabilityai/sd-turbo`.
3. Open **Generation Settings** and set:

   | Setting | Value |
   |---|---|
   | Inference Steps | 4 |
   | Image Height / Width | 512 / 512 |
   | Number of images to generate | 1 |
   | Guidance Scale | 1 |
   | Use seed | ticked |
   | Seed | 42 (then 11, 99 for more variants) |

   Leave **Use Safety Checker** and **Use Tiny AutoEncoder** unticked.
4. Open **Image to Image** and upload `sketch_filled_512.png`.
5. Paste a prompt from B2.
6. Set **Strength** to 0.8 (4 × 0.8 = 3 real steps).
7. Click **Generate**. The result is saved in `fastsdcpu\results`.
8. Know what to expect:
   - Strength 0.7 (2 real steps) keeps the skeleton but looks like the flat sketch, and the fire turned into a red blob in 8 of 12 test images.
   - Strength 0.8 gives more stylised results, but the skeleton was small or faint.
   - No-style wording gave the most keeps for this path.

## Path 2: SDXL-Turbo img2img (best results in the tests)
1. Under **Mode**, select **LCM**.
2. Open **Models** and set **LCM model** to `stabilityai/sdxl-turbo`.
3. Open **Generation Settings** and set:

   | Setting | Value |
   |---|---|
   | Inference Steps | 4 |
   | Image Height / Width | 512 / 512 |
   | Guidance Scale | 1 |
   | Use seed | ticked |
   | Seed | 42 (then 11, 99, and at least 3 seeds in total) |

4. Open **Image to Image** and upload one of:
   - `sketch_filled_512.png` for a night scene (all 3 tested seeds gave one clear skeleton).
   - `sketch_plain_512.png` for a pale daytime look and the best impressionist results.
5. Paste a prompt from B2.
6. Set **Strength** to 0.8. Strength 0.9 gives the identical image at 4 steps, so do not bother with it. Strength 0.7 stays close to the flat sketch.
7. Click **Generate**. The first run is slow because 13 GB has to load into RAM.
8. Variant without a sketch: open **Text to Image** instead of **Image to Image**, paste the prompt, and keep the same settings. It had the highest keep rate (6 of 12) and the best style response, but the skeleton and moon position are not controlled.
9. Use several seeds. One seed in the plain-sketch test drew two skeletons in pixel-art style.

## Path 3: SD 1.5 + LCM-LoRA + scribble ControlNet (follows layout, rough look)
1. Under **Mode**, select **LCM-LoRA**.
2. Open **Models** and set:
   - **LCM LoRA model** to `latent-consistency/lcm-lora-sdv1-5`
   - **LCM LoRA base model** to `stable-diffusion-v1-5/stable-diffusion-v1-5` (added in A6, step 4). With `Lykon/dreamshaper-8` every style looked the same.
3. Open **Generation Settings** and set:

   | Setting | Value |
   |---|---|
   | Inference Steps | 6 |
   | Image Height / Width | 512 / 512 |
   | Guidance Scale | 1 |
   | Use seed | ticked |
   | Seed | 42 (then 11, 99) |

4. Open **ControlNet** and set:
   1. Tick **Enable ControlNet**.
   2. **ControlNet model**: `control_v11p_sd15_scribble_fp16`
   3. **ControlNet conditioning scale**: 0.7
   4. **Preprocessor**: Canny (this matches the tests: a Canny edge map of the plain sketch fed to the scribble model).
   5. **Control image**: upload `sketch_plain_512.png`.
5. Open **Text to Image**, paste a prompt from B2 and click **Generate**.
6. Know what to expect: skeleton, fire and moon were in place in all 12 test images, but trees have stick legs and there are odd chairs. Only 2 of 12 were kept. Text-only without ControlNet (same mode, ControlNet off) mostly drew no skeleton.

## Switching between paths
1. Change only the mode and the model(s) from steps 1–2 of the path, then re-check the Generation Settings table.

## Running offline later
1. Open **Generation Settings** and tick **Use locally cached model or downloaded model folder(offline)**.
2. If a model then fails to load, untick it once with internet on. The tests downloaded some files selectively, so a strict offline check can fail.

---

# PART D: Problems seen on this PC

| Symptom | Cause | Fix |
|---|---|---|
| Window closes silently at "Loading pipeline components 20%" | Not enough free RAM or virtual memory | Close Chrome, IntelliJ, virtual machines, restart the PC, then retry. Path 2 needs about 11 GB. |
| `MemoryError` | Same | Same |
| "Application Control policy has blocked this file" | Smart App Control blocks an unsigned native file | Uninstall the package the file belongs to (A4). Do not change the Windows setting. |
| "mat1 and mat2 must have the same dtype, but got Half and Float" | transformers 5 keeps fp16 for text encoders | Apply the patch in A5. |
| Download stops with "memory allocation failed" | `xet` downloader | Set `HF_HUB_DISABLE_XET=1` (A6). |
| ControlNet model missing from the list | Added after the app started | Restart with `start-webui.bat`. |
| Base model `stable-diffusion-v1-5/...` not in the dropdown | Not in the app's list file | Add the line from A6, step 4 and restart. |
| Results lost the patch after an update | `git pull` overwrote it | Redo A5. |
| `start.bat` / `start-webui.bat` not found | Command Prompt was in the wrong folder | `cd` into the `fastsdcpu` folder first. |

# Not covered
- All tests used one sketch and one scene, so these settings are not confirmed for other drawings.
- Per-image time and RAM for Path 3 were not recorded.
- The pixel-art LoRA and pixel-art base model suggestions from the earlier report were not tested.
