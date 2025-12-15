import pickle
import numpy as np
import matplotlib.pyplot as plt
import gudhi

# # ---- Load PI features ----
# with open("TDA_2_PImage_features.pkl", "rb") as fp:
#     features = np.array(pickle.load(fp))

# img = features[0]   # shape: (H, W, 4)


# birth_max = 15
# pers_max  = 15

# fig, axes = plt.subplots(1, 4, figsize=(8, 5))

# for c in range(4):
#     im = axes[c].imshow(
#         img[..., c],
#         cmap="inferno",
#         origin="lower",
#         extent=[0, birth_max, 0, pers_max],
#         interpolation="nearest",
#         aspect="equal"
#     )

#     axes[c].set_xlabel("Birth", fontsize=12)

#     if c == 0:
#         axes[c].set_ylabel("Persistence", fontsize=12)
     

# plt.tight_layout()
# plt.savefig("1a0t_PI.png", dpi=600, bbox_inches="tight")
# plt.show()


# ---- Load PD features ----
with open("TDA_2_PDiagram_features.pkl", "rb") as fp:
    diagrams_all = pickle.load(fp)

pd_i = diagrams_all[1]

fig, axes = plt.subplots(1, 4, figsize=(10, 3))

pd_list = [
    pd_i["hyd_bd"][1],
    pd_i["hyd_bd"][2],
    pd_i["gen_bd"][1],
    pd_i["gen_bd"][2],
]

for c, pd in enumerate(pd_list):
    gudhi.plot_persistence_diagram(pd, axes=axes[c])
    axes[c].set_title("")

    axes[c].set_xlabel("Birth")

    if c == 0:
        axes[c].set_ylabel("Death")

    else:
        axes[c].yaxis.label.set_visible(False)


plt.tight_layout()
plt.savefig("1a4q_PD.png", dpi=600, bbox_inches="tight")
plt.show()





