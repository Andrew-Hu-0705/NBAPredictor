"""Model sanity tests: does the underlying model reject malformed input, and
does it produce well-formed probabilities — independent of the API layer."""
import pytest

from predict import load_artifacts, predict_game

MATCHUPS = [("BOS", "MIA"), ("LAL", "GSW"), ("MIL", "PHI"), ("DEN", "PHX")]


@pytest.fixture(scope="module")
def artifacts():
    return load_artifacts()


@pytest.mark.parametrize("home,away", MATCHUPS)
def test_win_probabilities_are_valid(artifacts, home, away):
    model, explainer, feature_cols = artifacts
    result = predict_game(home, away, model=model, explainer=explainer, feature_cols=feature_cols)

    assert 0.0 <= result["home_win_prob"] <= 1.0
    assert 0.0 <= result["away_win_prob"] <= 1.0
    assert abs(result["home_win_prob"] + result["away_win_prob"] - 1.0) < 1e-6
    assert result["predicted_winner"] in (home, away)


@pytest.mark.parametrize("home,away", MATCHUPS)
def test_shap_output_matches_feature_count(artifacts, home, away):
    model, explainer, feature_cols = artifacts
    result = predict_game(home, away, model=model, explainer=explainer, feature_cols=feature_cols)

    assert len(result["shap_values"]) == len(feature_cols)
    assert len(result["feature_values"]) == len(feature_cols)


def test_rejects_unknown_team(artifacts):
    model, explainer, feature_cols = artifacts
    with pytest.raises(ValueError):
        predict_game("XXX", "MIA", model=model, explainer=explainer, feature_cols=feature_cols)


def test_rejects_unknown_away_team(artifacts):
    model, explainer, feature_cols = artifacts
    with pytest.raises(ValueError):
        predict_game("BOS", "ZZZ", model=model, explainer=explainer, feature_cols=feature_cols)
