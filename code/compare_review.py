"""Side-by-side comparison of the key cross-model numbers before and after the word review."""
import os

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
before = pd.read_csv(f"{ROOT}/cross_model_summary_prereview.csv").set_index("model")
after = pd.read_csv(f"{ROOT}/cross_model_summary.csv").set_index("model")
cols = {
    "english_success": "English labels: share flipped",
    "noncognate_translation_success": "Non-cognate translations: share flipped",
    "cognate_translation_success": "Cognate translations: share flipped",
    "pseudoword_slope": "Pseudoword slope (nats per unit similarity)",
    "tr_position": "Translation vs pseudoword, position-adjusted (nats)",
    "tr_position_p": "  p",
    "ff_position": "False friend vs pseudoword, position-adjusted (nats)",
    "ff_position_p": "  p",
}
rows = []
for m in [k for k in after.index if k in before.index]:
    for c, label in cols.items():
        rows.append({"model": m, "measure": label, "before": before.at[m, c], "after": after.at[m, c]})
out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
for m, g in out.groupby("model", sort=False):
    print(f"--- {m}")
    print(g[["measure", "before", "after"]].round(4).to_string(index=False))
extra = [c for c in ["tr_vs_ff", "tr_vs_ff_se", "tr_vs_ff_p", "ff_position_noother", "ff_position_noother_se", "ff_position_noother_p"] if c in after.columns]
print("\nAfter review, additional checks:")
print(after[extra].round(4).to_string())
