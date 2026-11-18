"""H2: GEE logistic regression of detection on prior surprisal, clustered by image."""
from __future__ import annotations

import pandas as pd


def fit_surprisal_gee(feats: pd.DataFrame, reference_solver: str = "unet"):
    """feats: detected feature rows with columns detected, solver, z_surprisal, cls,
    size, dprime, severity, z_texture, image_id. Returns the fitted result and the
    DPS x surprisal interaction (coef, se, p)."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    d = feats.copy()
    d["solver"] = pd.Categorical(d["solver"], categories=[reference_solver] +
                                 sorted(set(d["solver"]) - {reference_solver}))
    formula = ("detected ~ solver * z_surprisal + C(cls) + C(size) + C(dprime) + C(severity)"
               " + z_texture")
    model = smf.gee(formula, groups="image_id", data=d, family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable())
    res = model.fit()
    terms = [t for t in res.params.index if t.startswith("solver[") and t.endswith(":z_surprisal")]
    inter = {t: dict(coef=float(res.params[t]), se=float(res.bse[t]), p=float(res.pvalues[t])) for t in terms}
    return res, inter
