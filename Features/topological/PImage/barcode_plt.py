import pickle
import numpy as np
import matplotlib.pyplot as plt
import gudhi

np.random.seed(42)

with open("TDA_2_PDiagram_features.pkl", "rb") as fp:
    diagrams_all = pickle.load(fp)

pd_i = diagrams_all[1]

fig, axes = plt.subplots(1, 4, figsize=(10, 3))

pd_list = [
    (pd_i["hyd_bd"][1], 1, {1: "tab:blue"}),  # Hydro H1
    (pd_i["hyd_bd"][2], 2, {2: "tab:green"}),   # Hydro H2
    (pd_i["gen_bd"][1], 1, {1: "tab:blue"}),  # Gen H1
    (pd_i["gen_bd"][2], 2, {2: "tab:green"}),   # Gen H2
]

K = 150

for c, (pd_raw, dim, cmap_dict) in enumerate(pd_list):
    pd_raw = np.array(pd_raw)

    if len(pd_raw) > K:
        idx = np.random.choice(len(pd_raw), K, replace=False)
        pd_filtered = pd_raw[idx]
    else:
        pd_filtered = pd_raw

    persistence = [(dim, (b, d)) for b, d in pd_filtered]

    gudhi.plot_persistence_barcode(
        persistence=persistence,
        alpha=0.9,
        max_intervals=20000,
        inf_delta=0.1,
        legend=True,
        colormap=cmap_dict,     # ✅ dict expected by your GUDHI
        axes=axes[c],
        fontsize=16
    )

    axes[c].set_title("")
    axes[c].set_xlabel("Filtration Parameter", fontsize=14)


plt.tight_layout()
plt.savefig("1a4q_Barcode.png", dpi=600, bbox_inches="tight")
plt.show()


