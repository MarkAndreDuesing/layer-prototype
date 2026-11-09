"""Price-to-performance of text-to-image models: Pareto frontier over the Artificial Analysis arena.

Data: Artificial Analysis text-to-image arena (AA-Image-T2I v2.1), read 2026-11-05.
Each row: model | Elo | Elo confidence interval | USD per 1,000 images at 1024x1024 on the creator's API, default settings.
Source: https://artificialanalysis.ai/text-to-image/arena/leaderboard-text

Method (see ../../reports/Image generation models hosting and costs.md, section 4):
  1. Pareto frontier: a model is efficient only if no other model is both cheaper and higher-rated.
  2. Along the frontier, compare the Elo gained with the extra price of each step.
  Shown for contrast only: a fitted "market line" Elo = a + b*log2(price) and its residuals, and naive Elo/$.
  Both were rejected as metrics (R^2 of the line is 0.10; Elo/$ just rewards the cheapest model).

Run with any Python 3 that has numpy:
    python price_performance.py
Output is saved next to this file as price_performance_output.txt.
"""
import numpy as np

D = """GPT Image 2.5 Sunburst (max)|1223|10|210.7
GPT Image 2.5 Flare (max)|1207|10|210.7
GPT Image 2 (high)|1189|11|211.0
Nano Banana 2.1|1176|12|33.6
Grok Imagine Image 2.0|1168|12|60.0
MAI-Image-2.6|1167|12|38.9
Nano Banana 2|1140|10|67.0
Muse Image|1127|11|10.0
GPT Image 1.5 (high)|1123|11|133.0
MAI-Image-2.6-Flash|1120|12|19.5
FLUX 3 Image|1119|12|48.0
MAI-Image-2.5|1118|11|48.1
Nano Banana Pro|1118|10|134.0
MAI-Image-2.5-Pro|1114|10|108.5
Nano Banana 2 Lite|1107|10|33.6
Seedream 5.0 Pro|1101|11|90.0
Qwen-Image-3.0-Pro|1097|12|40.0
Qwen-Image-3.0|1072|12|30.0
Qwen-Image-2.1-Pro|1058|12|35.3
grok-imagine-image-quality|1049|9|50.0
FLUX 3 Image Klein|1043|12|9.0
Fibo 2.0|1040|12|32.0
Krea 2 Large|1034|9|60.0
Qwen Image 2.0 Pro|1030|11|75.0
MAI-Image-2.5-Flash|1029|11|20.0
Qwen-Image-2.1-Turbo|1028|12|16.0
Seedream 4.5|1027|12|40.0
Wan 2.6 Image|1026|12|30.0
FLUX.2 [flex]|1026|11|60.0
Recraft V4.1 Utility|1026|11|35.0
Ideogram 4.0 (Quality)|1025|10|100.0
Luma UNI 1 Max|1024|12|100.0
grok-imagine-image|1022|11|20.0
FLUX.2 [max]|1021|10|70.0
Krea 2 Medium Turbo|1021|11|15.0
Seedream 5.0 Lite|1019|12|35.0
Recraft V4.1 Utility Pro|1018|9|210.0
HiDream-O1-Image-1.5|1017|11|80.0
Ideogram 4.5 (High)|1016|15|100.0
Wan2.6 Text to Image|1015|12|30.0
GPT Image 1 (high)|1014|12|167.0
Krea 2 Medium|1012|9|30.0
FLUX.2 [pro]|1003|10|30.0
Ideogram 4.0|1002|10|60.0
FLUX.2 [dev]|1000|0|12.0
Ming-Image-0.1-Design|999|9|30.0
P-Image-Ideogram (High)|997|14|15.0
FLUX.2 [dev] Turbo|995|11|8.0
Luma UNI 1|993|12|40.4
P-Image-Ideogram (Very High)|992|15|33.0
Qwen Image 2.0|990|16|35.0
FLUX.2 [dev] Flash|988|12|5.0
HunyuanImage 3.0 (Tencent Cloud)|988|16|32.0
Wan 2.7|987|12|26.0
Cosmos3-Super-Text2Image|986|10|40.0
Nano Banana (Gemini 2.5 Flash Image)|985|12|39.0
Ideogram 4.0 Instant|983|12|7.5
Ideogram 4.0 Fast (Quality)|982|12|17.5
Wan 2.7 Pro|979|12|75.0
HiDream-O1-Image|975|10|40.0
P-Image-Ideogram (Medium)|974|14|10.0
HunyuanImage 3.5 (Preview)|970|12|24.0
Ideogram 4.0 Fast|963|12|10.5
P-Image-Ideogram (Low)|961|14|7.5
Qwen Image Plus 2601|953|16|30.0
Recraft V4.1 Flash|949|12|7.0
Z-Image Turbo|936|11|5.0
GPT Image 1 Mini (medium)|934|17|11.0
Dreamina 3.1|932|17|30.0
FLUX.2 [klein] 9B|930|10|15.0
FLUX.1 Kontext [max]|929|12|80.0
Kling Image 3.0 Omni|919|16|28.0
FLUX.1 Kontext [pro]|918|18|40.0
ERNIE Image Turbo|915|10|10.5
Ideogram 3.0|910|17|60.0
P-Image-Ideogram (Very Low)|909|15|3.0
ERNIE Image|907|10|31.5
Qwen Image|904|18|20.0
P-Image|898|18|5.0
Vidu Q2|895|17|30.0
Z-Image Base|891|17|10.0
FLUX.1 Krea [dev]|889|38|25.0
GLM-Image|885|18|50.0
Lucid Origin Ultra|878|17|86.7
FLUX1.1 [pro]|871|19|40.0
Lucid Origin Fast|861|17|17.9
LongCat Image|857|19|130.0
FLUX1.1 [pro] Ultra|854|19|60.0
FIBO|850|19|40.0
Luma Photon|844|19|19.0
FLUX.2 [klein] 4B|838|12|14.0
Phoenix 0.9 Ultra|833|20|76.2
Phoenix 1.0 Fast|826|20|19.4
SRPO|826|20|26.0
FLUX.1 [dev]|826|43|25.0
FIBO Lite|813|16|20.0
HiDream-O1-Image-Dev|809|20|5.0
Luma Photon Flash|770|23|5.0
FLUX.1 [schnell]|740|20|3.0
OmniGen V2|714|26|150.0
Bagel|594|35|100.0"""

rows = []
for line in D.splitlines():
    n, e, c, p = line.split("|")
    rows.append((n, int(e), int(c), float(p)))
N = len(rows)
elo = np.array([r[1] for r in rows], float)
ci = np.array([r[2] for r in rows], float)
price = np.array([r[3] for r in rows], float)
lp = np.log2(price)

# Market line (shown for contrast, not used as the metric): Elo = a + b*log2(price)
b, a = np.polyfit(lp, elo, 1)
pred = a + b * lp
resid = elo - pred
r2 = 1 - ((elo - pred) ** 2).sum() / ((elo - elo.mean()) ** 2).sum()

# Pareto frontier: no other model has >= Elo and <= price, strictly better on one
pareto = []
for i in range(N):
    dominated = any(
        (elo[j] >= elo[i] and price[j] <= price[i]) and (elo[j] > elo[i] or price[j] < price[i])
        for j in range(N)
    )
    pareto.append(not dominated)

# Elo win probability against the leader
win_lead = 1 / (1 + 10 ** (-(elo - elo.max()) / 400))

print(f"N={N}  market slope b={b:.1f} Elo per price doubling  intercept={a:.0f}  R^2={r2:.2f}")
print(f"residual sd={resid.std():.0f}")
order = np.argsort(-resid)
print("\nTop 15 by value residual (Elo above market line at its price):")
print(f"{'model':38s} {'Elo':>5} {'$/1k':>7} {'pred':>5} {'resid':>6} {'CI':>4} {'pareto':>6} {'win v lead':>10}")
for i in order[:15]:
    print(f"{rows[i][0]:38s} {elo[i]:5.0f} {price[i]:7.1f} {pred[i]:5.0f} {resid[i]:+6.0f} {ci[i]:4.0f} "
          f"{str(pareto[i]):>6} {win_lead[i]:10.2f}")
print("\nPareto frontier (cheapest -> best):")
for i in sorted([i for i in range(N) if pareto[i]], key=lambda i: price[i]):
    print(f"{rows[i][0]:38s} {elo[i]:5.0f} {price[i]:7.1f} resid {resid[i]:+.0f}  win v lead {win_lead[i]:.2f}")
print("\nBottom 5 residual:")
for i in order[-5:]:
    print(f"{rows[i][0]:38s} {elo[i]:5.0f} {price[i]:7.1f} resid {resid[i]:+.0f}")
naive = elo / price
print("\nNaive Elo/$ top 5:", [rows[i][0] for i in np.argsort(-naive)[:5]])
