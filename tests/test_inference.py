import pandas as pd

from vaultech_analysis.inference import Predictor


def test_predict_returns_valid_result():
    predictor = Predictor()

    result = predictor.predict(
        die_matrix=5052,
        lifetime_2nd_strike_s=18.3,
        oee_cycle_time_s=13.5,
    )

    assert "predicted_bath_time_s" in result
    assert isinstance(result["predicted_bath_time_s"], float)
    assert result["predicted_bath_time_s"] > 0
    assert result["die_matrix"] == 5052
    assert result["lifetime_2nd_strike_s"] == 18.3
    assert result["oee_cycle_time_s"] == 13.5
    assert "model_metrics" in result


def test_predict_uses_default_oee_when_missing():
    predictor = Predictor()

    result = predictor.predict(
        die_matrix=5052,
        lifetime_2nd_strike_s=18.3,
        oee_cycle_time_s=None,
    )

    assert "predicted_bath_time_s" in result
    assert isinstance(result["predicted_bath_time_s"], float)
    assert result["predicted_bath_time_s"] > 0
    assert result["oee_cycle_time_s"] == predictor.default_oee


def test_predict_returns_error_for_unknown_die_matrix():
    predictor = Predictor()

    result = predictor.predict(
        die_matrix=9999,
        lifetime_2nd_strike_s=18.3,
        oee_cycle_time_s=13.5,
    )

    assert "error" in result
    assert "Unknown die_matrix" in result["error"]


def test_predict_batch_returns_series():
    predictor = Predictor()

    batch = pd.DataFrame([
        {
            "die_matrix": 5052,
            "lifetime_2nd_strike_s": 18.3,
            "oee_cycle_time_s": 13.5,
        },
        {
            "die_matrix": 5091,
            "lifetime_2nd_strike_s": 19.1,
            "oee_cycle_time_s": None,
        },
    ])

    preds = predictor.predict_batch(batch)

    assert isinstance(preds, pd.Series)
    assert len(preds) == 2
    assert preds.name == "predicted_bath_time_s"
    assert (preds > 0).all()