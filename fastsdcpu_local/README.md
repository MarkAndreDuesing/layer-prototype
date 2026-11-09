# Local changes to FastSD CPU

[FastSD CPU](https://github.com/rupeshs/fastsdcpu) is not stored here. The sketch-to-image tests used it at commit `85b2d79b3c075e6b0f2d1d9ee0cf4a1638b5fed8` with these local changes:

- `fastsdcpu-local.patch`:
  - `src/backend/lcm_text_to_image.py` casts the pipeline components to float32 on CPU. This fixes SDXL-Turbo's "Half and Float" error with transformers 5.
  - Two config files list the extra models that were tested.
- `start-webui-offline.bat`: starts the web UI without network access, using models already downloaded.

Apply them:

1. Clone FastSD CPU next to this folder at the tested commit:
   ```bash
   git clone https://github.com/rupeshs/fastsdcpu.git && cd fastsdcpu && git checkout 85b2d79b3c075e6b0f2d1d9ee0cf4a1638b5fed8
   ```
2. Apply the patch and copy the starter script:
   ```bash
   git apply ../fastsdcpu_local/fastsdcpu-local.patch && cp ../fastsdcpu_local/start-webui-offline.bat .
   ```
3. Follow [../SETUP_AND_USAGE_GUIDE.md](../SETUP_AND_USAGE_GUIDE.md) for the environment, models and the 3 tested paths. Note that `git pull` would overwrite the patch.
