from typing import List

from pydantic import BaseModel, field_validator, model_validator

from teams import NBA_TEAMS

_TEAM_SET = set(NBA_TEAMS)


class PredictRequest(BaseModel):
    home_team: str
    away_team: str

    @field_validator("home_team", "away_team")
    @classmethod
    def team_must_be_valid(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in _TEAM_SET:
            raise ValueError(f"'{v}' is not a recognized NBA team abbreviation")
        return v

    @model_validator(mode="after")
    def teams_must_differ(self):
        if self.home_team == self.away_team:
            raise ValueError("home_team and away_team must be different")
        return self


class ShapFactor(BaseModel):
    feature: str
    value: float
    shap_value: float
    favors: str  # "home" or "away"


class PredictResponse(BaseModel):
    home_team: str
    away_team: str
    home_win_prob: float
    away_win_prob: float
    predicted_winner: str
    confidence: float
    shap_explanation: List[ShapFactor]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    features_loaded: bool
    model_source: str
