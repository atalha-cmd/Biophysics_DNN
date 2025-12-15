def PrepareImage(pro_PH_filepath, pro_hydro_PH_filepath, npy_filepath):
    import numpy as np
    import gudhi.representations as gr  

    rs   = 0.25
    b_thr = 15
    p_thr = 15
    d_thr = b_thr + p_thr
    small = 1e-4
    nx = int(np.ceil(b_thr / rs))
    ny = int(np.ceil(p_thr / rs))
    resolution = (nx, ny)

    ProPHFile      = pro_PH_filepath
    ProPHHydroFile = pro_hydro_PH_filepath
    OutFile        = npy_filepath

    # -------- load (b, d) for dims 1 and 2 ----------
    def load_bd(path):
        out = {1: [], 2: []}
        with open(path, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 3:
                    continue
                dim, b, d = int(parts[0]), float(parts[1]), float(parts[2])

                if dim in (1, 2):
                    p = d - b
                    if p >= small and b <= b_thr and p <= p_thr and d <= d_thr:
                        out[dim].append([b, d])

        # convert lists to arrays
        return {k: np.array(v, float) for k, v in out.items()}

    hyd_bd = load_bd(ProPHHydroFile)  # (birth, death) hydrophobic
    gen_bd = load_bd(ProPHFile)       # (birth, death) general

    # -------- convert to (birth, persistence) ----------
    def bd_to_bp(diag_bd):
        diag_bd = np.asarray(diag_bd, float)
        if diag_bd.size == 0:
            return np.zeros((0, 2), dtype=float)
        b = diag_bd[:, 0]
        p = diag_bd[:, 1] - diag_bd[:, 0]
        p[p < 0] = 0.0
        return np.stack([b, p], axis=1)

    hyd_bp = {k: bd_to_bp(v) for k, v in hyd_bd.items()}
    gen_bp = {k: bd_to_bp(v) for k, v in gen_bd.items()}

    # -------- GUDHI PersistenceImage ----------
    def weight_exp_soft(bp, tau=2.0):  
        b, p = float(bp[0]), float(bp[1])
        return np.exp(p / tau)

    def weight_persistence(bp):
        b, p = float(bp[0]), float(bp[1])
        return max(p, 0.0)

    def weight_log(bp):
        b, p = float(bp[0]), float(bp[1])
        return np.log1p(max(p, 0.0))

    def weight_sqrt(bp):
        b, p = float(bp[0]), float(bp[1])
        return np.sqrt(max(p, 0.0))

    def weight_exp(bp):
        b, p = float(bp[0]), float(bp[1])
        return np.exp(max(p, 0.0))  # exp(p)

    def weight_inv(bp):
        b, p = float(bp[0]), float(bp[1])
        return 1.0 / (p + 1e-6)

    PI = gr.PersistenceImage(
        bandwidth=0.4,
        weight=weight_exp,
        resolution=resolution,
        im_range=(0.0, b_thr, 0.0, p_thr)  # (bmin, bmax, pmin, pmax)
    )

    def to_img(bp):
        bp = np.asarray(bp, float)
        if bp.size == 0:
            return np.zeros((ny, nx), dtype=np.float32)
        vec = PI.fit_transform([bp])[0]
        return vec.reshape(ny, nx).astype(np.float32)

    hydro_h1 = to_img(hyd_bp.get(1, np.zeros((0, 2))))
    hydro_h2 = to_img(hyd_bp.get(2, np.zeros((0, 2))))
    gen_h1   = to_img(gen_bp.get(1, np.zeros((0, 2))))
    gen_h2   = to_img(gen_bp.get(2, np.zeros((0, 2))))

    feature = np.stack([hydro_h1, hydro_h2, gen_h1, gen_h2], axis=-1)

    # ---- Save persistence image ----
    with open(OutFile, "wb") as outfile:
        np.save(outfile, feature)

    # --------
    diagrams = {
        "hyd_bd": hyd_bd,   # (birth, death) hydrophobic H1/H2
        "gen_bd": gen_bd,   # (birth, death) general H1/H2
        "hyd_bp": hyd_bp,   # (birth, persistence) hydrophobic H1/H2
        "gen_bp": gen_bp    # (birth, persistence) general H1/H2
    }

    # Return both PI and PD info
    return feature, diagrams
