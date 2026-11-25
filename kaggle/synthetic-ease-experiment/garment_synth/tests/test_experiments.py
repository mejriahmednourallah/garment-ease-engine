"""Tests for the SYNTHETIC-data experiment code: leakage guards, split hygiene and metric sanity."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fit_experiments as fe
import generate_synthetic_garments as gen


@pytest.fixture(scope="module")
def rows_and_parts(data_dir):
    main = fe.load_set(data_dir / "main", True)
    R = fe.build_rows(main)
    R["partition"] = R.customer_id.map(main["splits_customers"].set_index("customer_id").partition)
    hold = main["splits_garments"].set_index("garment_id")
    R["unseen_garment"] = R.garment_id.map(hold.holdout_unseen_garment).astype(int)
    return main, R


@pytest.fixture(scope="module")
def results(data_dir, tmp_path_factory):
    return fe.run(data_dir, tmp_path_factory.mktemp("results"), seed=0)


# ---------------------------------------------------------------- leakage guards
def test_feature_columns_exclude_labels_latents_ids_and_metadata(sets):
    latent_cols = {c for k in sets for n in sets[k] if n.startswith("latent_") for c in sets[k][n].columns}
    for col in fe.FEATURE_SOURCE_COLUMNS:
        assert not fe.is_forbidden_feature(col), col
        assert col not in fe.LABEL_COLUMNS and col not in latent_cols
    assert "fit_preference" in fe.FEATURE_SOURCE_COLUMNS  # stated preference is a legitimate feature


def test_features_do_not_change_when_labels_ids_latents_are_perturbed(rows_and_parts):
    _, R = rows_and_parts
    base = fe.build_features(R)
    R2 = R.copy()
    rng = np.random.default_rng(0)
    for c in fe.LABEL_COLUMNS + ["customer_id", "garment_id", "observation_id", "sim_scenario", "partition", "stress_cohort"]:
        if c in R2:
            R2[c] = rng.permutation(R2[c].to_numpy())
    pert = fe.build_features(R2)
    for k in base:
        assert np.array_equal(base[k], pert[k], equal_nan=True), k


def test_label_permutation_control_collapses(results):
    real = results["A_customer_split_test"]["logreg"]["pair"]["top1_incl_none"]
    shuffled = results["label_permutation_control"]["logreg_top1_incl_none"]
    assert shuffled < real - 0.05


# ---------------------------------------------------------------- split hygiene / train-only preprocessing
def test_customer_partitions_disjoint_in_every_experiment_split(rows_and_parts):
    _, R = rows_and_parts
    tr, te = R.partition == "train", R.partition == "test"
    assert set(R.customer_id[tr]).isdisjoint(R.customer_id[te])
    assert set(R.customer_id[R.partition == "val"]).isdisjoint(R.customer_id[tr] if tr.any() else [])
    # unseen-garment: train garments exclude held-out garments; test customers x held-out garments
    a, b = R[tr & (R.unseen_garment == 0)], R[te & (R.unseen_garment == 1)]
    assert set(a.garment_id).isdisjoint(b.garment_id) and set(a.customer_id).isdisjoint(b.customer_id)
    # leave-one-style-out
    for st in gen.STYLES:
        a, b = R[tr & (R.garment_style != st)], R[te & (R.garment_style == st)]
        assert set(a.garment_id).isdisjoint(b.garment_id) and set(a.customer_id).isdisjoint(b.customer_id)


def test_preprocessing_statistics_come_from_training_rows_only(rows_and_parts):
    _, R = rows_and_parts
    tr = R[R.partition == "train"].reset_index(drop=True)
    F = fe.build_features(tr)
    prep = fe.Prep().fit(F)
    ok = F["app"] & ~np.isnan(F["ease"])
    manual = [np.median(F["ease"][ok[:, j], j]) for j in range(3)]
    assert np.allclose(prep.ease_med, manual)
    P = fe.Pairs(tr)
    lr = fe.LogReg(C=0.3, lam=1.0).fit(tr, F, P)
    X, _ = prep.X_row(F)
    assert np.allclose(lr.acc[0].mean_, X.mean(0))      # scaler fitted on train rows only
    assert np.allclose(lr.choice_sc.mean_, X.mean(0))


def test_pairs_helper_is_consistent(rows_and_parts):
    _, R = rows_and_parts
    r = R[R.partition == "test"].reset_index(drop=True)
    P = fe.Pairs(r)
    assert P.counts.sum() == len(r) and (P.truth[P.truth >= 0] >= 0).all()
    chosen = P.argmax(np.random.default_rng(0).random(len(r)))
    assert ((chosen >= P.starts) & (chosen < P.starts + P.counts)).all()
    assert (P.argmax(np.zeros(len(r)), np.zeros(len(r), bool)) == -1).all()


# ---------------------------------------------------------------- metric / output sanity
def test_all_methods_and_metrics_reported_for_every_split(results):
    methods = {"majority", "fixed_size_M", "weighted_ease_prior", "weighted_ease_tuned", "logreg"}
    for block in ["A_customer_split_test", "B_unseen_garment_test", "C_unseen_style_pooled_leave_one_style_out",
                  "D_unseen_scenario_S3_test", "E_stress_overall"]:
        assert set(results[block]) == methods, block
        for m, v in results[block].items():
            p = v["pair"]
            for k in ("top1_incl_none", "top1_when_acceptable_exists"):
                assert 0 <= p[k] <= 1
            assert {"precision", "recall", "f1", "support"} <= set(p["none_acceptable"])
            lo, hi = p["top1_incl_none_ci95_by_customer"]
            assert lo <= p["top1_incl_none"] <= hi or abs(p["top1_incl_none"] - lo) < 0.05
            if m != "fixed_size_M":
                for d in gen.DIMS:
                    assert 0 <= v["regional"][d]["accuracy"] <= 1 and set(v["regional"][d]["recall"]) == set(gen.FIT_LABELS)
    for m in ("weighted_ease_prior", "weighted_ease_tuned", "logreg"):
        for k in (2, 3):
            assert results["A_customer_split_test"][m]["pair"][f"top{k}_when_acceptable_exists"] is not None


def test_labels_are_not_trivially_recovered_by_baseline_formula(results):
    prior = results["A_customer_split_test"]["weighted_ease_prior"]["pair"]["top1_incl_none"]
    tuned = results["A_customer_split_test"]["weighted_ease_tuned"]["pair"]["top1_incl_none"]
    assert prior < 0.8 and tuned < 0.85


def test_learned_model_beats_trivial_baselines_in_distribution(results):
    A = results["A_customer_split_test"]
    lr = A["logreg"]["pair"]["top1_incl_none"]
    assert lr > A["fixed_size_M"]["pair"]["top1_incl_none"] and lr > A["majority"]["pair"]["top1_incl_none"]


def test_stress_results_cover_missing_data_cohorts_and_predictions_are_finite(results, data_dir):
    by = results["E_stress_by_cohort"]["customer_cohort"]
    assert {"all_body_missing", "missing_chest", "extreme_size_large", "ood_population", "control"} <= set(by)
    for coh, v in by.items():
        for m, x in v.items():
            assert np.isfinite(x["top1_incl_none"])
    stress = fe.load_set(data_dir / "stress")
    Rs = fe.build_rows(stress)
    F = fe.build_features(Rs)
    assert np.isnan(F["ease"][F["app"]]).any()          # missingness really reaches the model code path


def test_synthetic_notice_and_interpretability_outputs(results):
    assert "SYNTHETIC" in results["synthetic_notice"]
    coefs = results["logreg_top_coefficients"]
    assert len(coefs["overall_acceptable_logreg_std_coef"]) > 5 and len(coefs["size_choice_logit_std_coef"]) > 5
    assert results["config"]["theta"] in (0.3, 0.4, 0.5, 0.6, 0.7)
