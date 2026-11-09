Final chosen configuration per path (source folders are relative to experiments/)

path1_sdturbo_img2img
  config: filled_s80 from r2_p1/seed*/
  seeds: [123123, 7, 2024]
  images: final/path1_sdturbo_img2img/
  sheet: final/FINAL_path1_sdturbo_img2img.png
  note: Style-capable setting, but the skeleton is lost or deformed in all 3 seeds. Strength 0.7 (filled_s70) keeps the skeleton but barely restyles; see r2_p1.

path2_sdxlturbo_img2img
  config: filled_s80 from r2_p2/seed*/
  seeds: [123123, 7, 2024]
  images: final/path2_sdxlturbo_img2img/
  sheet: final/FINAL_path2_sdxlturbo_img2img.png
  note: Recommended overall. Seed 2024 duplicates the skeleton.

path3_controlnet_neutral
  config: neutral_g10_c70 from r2_p3/seed*/
  seeds: [123123, 7]
  images: final/path3_controlnet_neutral/
  sheet: final/FINAL_path3_controlnet_neutral.png
  note: Best layout adherence; only 2 seeds were generated (123123, 7). 6-step variant: neutral_g10_c70_6steps in the same folders.
