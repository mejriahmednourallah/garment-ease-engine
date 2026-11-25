"""Validation tests for the SYNTHETIC garment-ease generator."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import generate_synthetic_garments as gen

MODEL_VISIBLE = ["customers", "garments", "garment_sizes"]


# ---------------------------------------------------------------- reproducibility / configuration
def test_same_seed_is_byte_identical(tmp_path, cfg):
    a, b = tmp_path / "a", tmp_path / "b"
    _, _, ma = gen.write_all(cfg, a)
    _, _, mb = gen.write_all(cfg, b)
    assert {k: v["sha256"] for k, v in ma["files"].items()} == {k: v["sha256"] for k, v in mb["files"].items()}


def test_different_seed_changes_data(tmp_path, cfg):
    other = gen.GenConfig(**{**gen.asdict(cfg), "seed": cfg.seed + 1})
    _, _, m1 = gen.write_all(cfg, tmp_path / "a")
    _, _, m2 = gen.write_all(other, tmp_path / "b")
    assert m1["files"]["main/customers.csv"]["sha256"] != m2["files"]["main/customers.csv"]["sha256"]


def test_main_data_independent_of_stress_size(tmp_path, cfg):
    bigger = gen.GenConfig(**{**gen.asdict(cfg), "stress_customers": cfg.stress_customers + 50})
    _, _, m1 = gen.write_all(cfg, tmp_path / "a")
    _, _, m2 = gen.write_all(bigger, tmp_path / "b")
    assert m1["files"]["main/observations.csv"]["sha256"] == m2["files"]["main/observations.csv"]["sha256"]


def test_configurable_sizes(sets, cfg):
    m, s = sets["main"], sets["stress"]
    assert len(m["customers"]) == cfg.n_customers and len(m["garments"]) == cfg.n_garments
    assert len(s["customers"]) == cfg.stress_customers and len(s["garments"]) == cfg.stress_garments
    assert m["observations"]["pair_id"].nunique() == cfg.n_customers * cfg.pairs_per_customer


def test_manifest_hashes_match(data_dir):
    man = json.loads((data_dir / "MANIFEST.json").read_text())
    for rel, info in man["files"].items():
        assert hashlib.sha256((data_dir / rel).read_bytes()).hexdigest() == info["sha256"]


# ---------------------------------------------------------------- SYNTHETIC marking and documentation
def test_every_csv_marked_synthetic(sets, data_dir):
    for kind in sets:
        for name, df in sets[kind].items():
            assert df.columns[0] == "synthetic_marker" and (df["synthetic_marker"] == "SYNTHETIC").all(), name
    assert "SYNTHETIC" in (data_dir / "README_SYNTHETIC.txt").read_text()
    assert "SYNTHETIC" in (data_dir / "DATA_DICTIONARY.md").read_text()


def test_columns_match_data_dictionary(sets):
    for kind in sets:
        for name, df in sets[kind].items():
            assert list(df.columns) == [c[0] for c in gen.COLUMNS[name]], (kind, name)


def test_assumptions_registry_marks_everything_unvalidated(data_dir):
    items = json.loads((data_dir / "assumptions.json").read_text())
    assert items and all("UNVALIDATED" in i["status"] for i in items)


# ---------------------------------------------------------------- body measurements
def test_body_values_positive_plausible_main(sets):
    c, lc = sets["main"]["customers"], sets["main"]["latent_customers"]
    for d in gen.DIMS:
        lo, hi = gen.BODY_BOUNDS[d]
        assert lc[f"true_{d}_cm"].between(lo, hi).all()
        assert c[f"{d}_cm"].notna().all() and (c[f"{d}_cm"] > 30).all() and (c[f"{d}_cm"] < 200).all()
    assert (lc.true_waist_cm / lc.true_chest_cm).between(*gen.RATIO_MAIN["waist_chest"]).all()
    assert (lc.true_hip_cm / lc.true_chest_cm).between(*gen.RATIO_MAIN["hip_chest"]).all()


def test_body_measurements_are_correlated_not_independent(sets):
    lc = sets["main"]["latent_customers"]
    corr = lc[["true_chest_cm", "true_waist_cm", "true_hip_cm"]].corr().to_numpy()
    assert corr[0, 1] > 0.6 and corr[0, 2] > 0.5 and corr[1, 2] > 0.5


def test_regional_variation_present(sets):
    c = sets["main"]["customers"]
    means = c.groupby("sim_region")["chest_cm"].mean()
    assert means.max() - means.min() > 3 and c.sim_region.nunique() == 5


def test_measurement_noise_between_reported_and_true(sets):
    c = sets["main"]["customers"].merge(sets["main"]["latent_customers"], on="customer_id")
    err = c.chest_cm - c.true_chest_cm
    assert 0.5 < err.std() < 1.6 and abs(err.mean()) < 0.3


def test_customer_ids_carry_no_feature_information(sets):
    c = sets["main"]["customers"]
    rank = np.arange(len(c))
    for col in ["chest_cm", "waist_cm", "hip_cm"]:
        assert abs(pd.Series(rank).corr(c[col], method="spearman")) < 0.15
    s = sets["stress"]["customers"]
    half = len(s) // 2
    assert abs((s.stress_cohort[:half] == "ood_population").mean() - (s.stress_cohort[half:] == "ood_population").mean()) < 0.15


# ---------------------------------------------------------------- garments and size charts
def test_size_charts_ordered_and_graded_main(sets):
    g, s = sets["main"]["garments"], sets["main"]["garment_sizes"]
    order = {l: i for i, l in enumerate(gen.SIZE_ORDER)}
    assert (s.size_label.map(order) == s.size_rank).all()
    for gid, grp in s.groupby("garment_id"):
        assert (np.diff(grp.size_rank) == 1).all() and "M" in set(grp.size_label)
        for d in gen.DIMS:
            for pre in ("garment", "label_body"):
                col = grp[f"{pre}_{d}_cm"].dropna()
                assert (np.diff(col) > 0).all(), (gid, pre, d)
    assert (s.groupby("garment_id").size().to_numpy() == g.set_index("garment_id").loc[s.garment_id.unique(), "n_sizes"].to_numpy()).all()


def test_garment_measure_is_not_the_label_body_measure(sets):
    s = sets["main"]["garment_sizes"]
    diff = (s.garment_waist_cm - s.label_body_waist_cm).dropna()
    assert diff.std() > 1 and diff.abs().mean() > 0.5


def test_garment_values_positive_and_plausible(sets):
    for kind in sets:
        lat = sets[kind]["latent_garment_sizes"]
        s = sets[kind]["garment_sizes"]
        for d in gen.DIMS:
            for df, col in ((lat, f"actual_garment_{d}_cm"), (s, f"garment_{d}_cm"), (s, f"label_body_{d}_cm")):
                v = df[col].dropna()
                assert (v > 20).all() and (v < 260).all(), (kind, col)


def test_measured_dims_consistent_with_nan_and_labels(sets):
    for kind in sets:
        g = sets[kind]["garments"].set_index("garment_id")
        s = sets[kind]["garment_sizes"].merge(g[["measured_dims"]], left_on="garment_id", right_index=True)
        o = sets[kind]["observations"].merge(g[["measured_dims"]], left_on="garment_id", right_index=True)
        for d in gen.DIMS:
            applicable = s.measured_dims.str.split("|").apply(lambda x: d in x)
            assert s.loc[~applicable, f"garment_{d}_cm"].isna().all()
            assert s.loc[~applicable, f"label_body_{d}_cm"].isna().all()
            app_o = o.measured_dims.str.split("|").apply(lambda x: d in x)
            assert (o.loc[~app_o, f"fit_{d}"] == "not_applicable").all()
            assert (o.loc[app_o, f"fit_{d}"] != "not_applicable").all()


def test_stretch_documented_and_not_deterministic_of_usable(sets):
    g = sets["main"]["garments"]
    lg = sets["main"]["latent_garments"]
    assert (g.stretch_direction == "crosswise").all() and (g.stretch_crosswise_pct >= 0).all()
    assert g.stretch_crosswise_pct.max() <= 100
    assert abs(lg.usable_stretch_fraction.corr(lg.true_stretch_crosswise_pct)) < 0.3
    knit = g[g.fabric_class == "knit"].stretch_crosswise_pct
    woven = g[g.fabric_class == "woven"].stretch_crosswise_pct
    assert knit.mean() > woven.mean() + 10
    col = [c for c in gen.COLUMNS["garments"] if c[0] == "stretch_crosswise_pct"][0]
    assert col[3] == "%" and "NOT on-body" in col[4] and "crosswise" in col[4] and "fixed lab load" in col[4]


# ---------------------------------------------------------------- keys, labels and preferred sizes
def test_keys_unique_and_foreign_keys_valid(sets):
    for kind in sets:
        d = sets[kind]
        assert d["customers"].customer_id.is_unique and d["garments"].garment_id.is_unique
        assert d["observations"].observation_id.is_unique
        assert not d["garment_sizes"].duplicated(["garment_id", "size_label"]).any()
        assert set(d["observations"].customer_id) <= set(d["customers"].customer_id)
        keys = set(zip(d["garment_sizes"].garment_id, d["garment_sizes"].size_label))
        assert set(zip(d["observations"].garment_id, d["observations"].size_label)) <= keys
        assert len(d["latent_observations"]) == len(d["observations"])
        assert (d["latent_observations"].observation_id == d["observations"].observation_id).all()


def test_label_domain_and_internal_consistency(sets):
    for kind in sets:
        o = sets[kind]["observations"]
        dom = {"tight", "acceptable", "loose", "not_applicable"}
        for d in gen.DIMS:
            assert set(o[f"fit_{d}"]) <= dom
        acc = np.all([(o[f"fit_{d}"].isin(["acceptable", "not_applicable"])) for d in gen.DIMS], axis=0)
        assert (acc.astype(int) == o.fit_overall_acceptable).all()
        grp = o.groupby("pair_id")
        assert (grp.is_preferred_size.sum() <= 1).all()
        assert (o.loc[o.is_preferred_size == 1, "fit_overall_acceptable"] == 1).all()
        none = grp.fit_overall_acceptable.sum() == 0
        assert (none.astype(int) == grp.pair_none_acceptable.first()).all()
        assert (grp.is_preferred_size.sum().eq(0) == none).all()
        pref = o[o.is_preferred_size == 1].set_index("pair_id").size_label
        pl = grp.pair_preferred_size.first()
        assert (pl[none] == "NONE").all() and (pl[~none].to_numpy() == pref.reindex(pl[~none].index).to_numpy()).all()


def test_class_balance_and_none_acceptable_present(sets):
    o = sets["main"]["observations"]
    for d in gen.DIMS:
        v = o.loc[o[f"fit_{d}"] != "not_applicable", f"fit_{d}"].value_counts(normalize=True)
        assert (v.reindex(["tight", "acceptable", "loose"]) > 0.10).all(), d
    rate = o.drop_duplicates("pair_id").pair_none_acceptable.mean()
    assert 0.05 < rate < 0.50
    pr = o.drop_duplicates("pair_id")
    assert pr.pair_preferred_size.nunique() >= 6


def test_multiple_candidate_sizes_per_pair(sets):
    n = sets["main"]["observations"].groupby("pair_id").size()
    assert n.min() >= 4 and n.max() <= 7


# ---------------------------------------------------------------- scenarios
def test_scenarios_main_vs_stress(sets):
    main_s = set(sets["main"]["latent_customers"].sim_scenario)
    stress_s = set(sets["stress"]["latent_customers"].sim_scenario)
    assert main_s == set(gen.SCENARIOS[:3]) and gen.SCENARIOS[3] in stress_s and gen.SCENARIOS[3] not in main_s


def test_labels_not_a_deterministic_function_of_baseline_formula(sets):
    """Weighted-ease penalty with ANY single desired-ease table cannot reproduce the preferred size: check agreement is far from 1."""
    d = sets["main"]
    o = d["observations"].merge(d["customers"], on="customer_id").merge(d["garment_sizes"], on=["garment_id", "size_label"])
    ease = np.column_stack([o[f"garment_{x}_cm"] - o[f"{x}_cm"] for x in gen.DIMS])
    des = o.fit_preference.map({"slim": 3, "regular": 6, "loose": 10}).to_numpy()[:, None]
    o["pen"] = np.nansum((ease - des) ** 2, axis=1)
    pick = o.loc[o.groupby("pair_id").pen.idxmin()].set_index("pair_id")
    truth = o.drop_duplicates("pair_id").set_index("pair_id").pair_preferred_size
    assert (pick.size_label.reindex(truth.index) == truth).mean() < 0.6


def test_latent_noise_and_customer_variation(sets):
    lc = sets["main"]["latent_customers"]
    assert lc.tol_tight_chest_cm.std() > 0.5 and lc.desired_offset_waist_cm.std() > 0.8
    assert (lc.tol_loose_chest_cm > lc.tol_tight_chest_cm).mean() > 0.8       # asymmetry on average
    assert (lc.behavioural_preference != sets["main"]["customers"].fit_preference).mean() > 0.04
    assert sets["main"]["latent_observations"][[f"label_flipped_{d}" for d in gen.DIMS]].to_numpy().sum() > 0


# ---------------------------------------------------------------- latent / feature separation
def test_no_latent_or_label_columns_in_model_visible_files(sets):
    banned = ("true_", "actual_", "latent", "usable", "rigidity", "tol_", "desired", "utility", "discomfort", "preferred", "fit_")
    for kind in sets:
        for name in MODEL_VISIBLE:
            for col in sets[kind][name].columns:
                assert not any(b in col for b in banned if not (b == "fit_" and col == "fit_preference")), (name, col)
    for name in ("latent_customers", "latent_garments", "latent_garment_sizes", "latent_observations"):
        assert name in sets["main"] and name in sets["stress"]


def test_dictionary_roles_separate_features_from_labels_and_latents():
    roles = {c: r for f, cols in gen.COLUMNS.items() for c, _, r, _, _ in cols if f in MODEL_VISIBLE}
    assert "label" not in roles.values() and "latent" not in roles.values()
    assert all(r == "latent" or r in ("marker", "key", "flag") for f, cols in gen.COLUMNS.items() if f.startswith("latent")
               for _, _, r, _, _ in cols)


# ---------------------------------------------------------------- splits
def test_customer_splits_disjoint_and_complete(sets):
    sp = sets["main"]["splits_customers"]
    c = sets["main"]["customers"]
    assert sp.customer_id.is_unique and set(sp.customer_id) == set(c.customer_id)
    frac = sp.partition.value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.03 and abs(frac["val"] - 0.15) < 0.03 and abs(frac["test"] - 0.15) < 0.03
    o = sets["main"]["observations"].merge(sp, on="customer_id")
    assert (o.groupby("customer_id").partition.nunique() == 1).all()


def test_unseen_garment_split(sets, cfg):
    sg = sets["main"]["splits_garments"]
    assert sg.holdout_unseen_garment.sum() == int(cfg.unseen_garment_fraction * cfg.n_garments)


# ---------------------------------------------------------------- stress set
def test_stress_set_separate_flagged_and_has_missing_and_edge_cases(sets, data_dir):
    assert (data_dir / "stress" / "customers.csv").exists() and "splits_customers" not in sets["stress"]
    c, g, s = sets["stress"]["customers"], sets["stress"]["garments"], sets["stress"]["garment_sizes"]
    assert c.customer_id.str.startswith("S").all() and g.garment_id.str.startswith("X").all()
    assert set(sets["main"]["customers"].stress_cohort) == {"none"}
    for col in ("chest_cm", "waist_cm", "hip_cm"):
        assert c[col].isna().mean() > 0.05
        assert sets["main"]["customers"][col].isna().sum() == 0
    assert g.stretch_crosswise_pct.isna().any() and (sets["stress"]["latent_garments"].true_stretch_crosswise_pct > 100).any()
    assert (c.stress_cohort == "all_body_missing").any() and {"extreme_size_small", "extreme_size_large", "extreme_proportion"} <= set(c.stress_cohort)
    assert {"sparse_chart", "missing_garment_measure", "very_loose_cut", "very_fitted_cut"} <= set(g.stress_cohort)
    assert g[g.stress_cohort == "sparse_chart"].n_sizes.eq(3).all()
    lc = sets["stress"]["latent_customers"]
    assert lc.true_chest_cm.min() < 75 and lc.true_chest_cm.max() > 130
    assert _applicable_missing(sets["stress"]) > 0 and _applicable_missing(sets["main"]) == 0


def _applicable_missing(d):
    """Garment-chart cells that the garment type measures but that are blank (true missingness, not 'n/a')."""
    s = d["garment_sizes"].merge(d["garments"][["garment_id", "measured_dims"]], on="garment_id")
    n = 0
    for dim in gen.DIMS:
        app = s.measured_dims.str.split("|").apply(lambda x: dim in x)
        n += int(s.loc[app, f"garment_{dim}_cm"].isna().sum())
    return n


def test_stress_positivity_still_holds(sets):
    lc = sets["stress"]["latent_customers"]
    for d in gen.DIMS:
        assert (lc[f"true_{d}_cm"] > 0).all()
