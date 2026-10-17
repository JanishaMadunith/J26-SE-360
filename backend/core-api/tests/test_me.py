def test_me_and_consent(client, user):
    me = client.get("/v1/me", headers=user["headers"]).json()
    assert me["consent_version"] is None and me["consented_at"] is None
    r = client.post("/v1/me/consent", json={"version": "2026-10-v1"}, headers=user["headers"])
    assert r.status_code == 200
    assert r.json()["consent_version"] == "2026-10-v1" and r.json()["consented_at"] is not None


def test_diet_profile_default_then_update(client, user):
    h = user["headers"]
    d = client.get("/v1/me/diet-profile", headers=h).json()
    assert d["vegetarian"] is False and d["allergies"] == []
    body = {
        "vegetarian": True,
        "vegan": False,
        "halal": False,
        "allergies": ["peanuts"],
        "dislikes": [],
        "health_conditions": [],
        "daily_calorie_target": 2000,
    }
    assert client.put("/v1/me/diet-profile", json=body, headers=h).status_code == 200
    assert client.get("/v1/me/diet-profile", headers=h).json()["allergies"] == ["peanuts"]
    body["daily_calorie_target"] = 100  # below the contract's minimum
    assert client.put("/v1/me/diet-profile", json=body, headers=h).status_code == 422
