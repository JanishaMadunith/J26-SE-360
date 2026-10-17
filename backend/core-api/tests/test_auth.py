from conftest import auth_headers, register


def test_register_returns_tokens_and_user(client):
    t = register(client, "Alice@Example.com")
    assert t["token_type"] == "bearer" and t["access_token"] and t["refresh_token"]
    assert t["user"]["email"] == "alice@example.com"  # normalised
    assert t["user"]["timezone"] == "Asia/Colombo"
    assert t["expires_in"] == 3600


def test_duplicate_email_rejected(client):
    register(client, "bob@example.com")
    r = client.post(
        "/v1/auth/register", json={"email": "BOB@example.com", "password": "another-pass-1", "display_name": "Bob"}
    )
    assert r.status_code == 409


def test_register_validation(client):
    bad = [
        {"email": "not-an-email", "password": "longenough1", "display_name": "x"},
        {"email": "a@b.co", "password": "short", "display_name": "x"},
        {"email": "a@b.co", "password": "longenough1", "display_name": "x", "timezone": "Mars/Base"},
        {"email": "a@b.co", "password": "longenough1", "display_name": "x", "is_admin": True},
    ]
    for body in bad:
        assert client.post("/v1/auth/register", json=body).status_code == 422, body


def test_login_and_wrong_password(client):
    register(client, "carol@example.com", "s3cret-pass!")
    ok = client.post("/v1/auth/login", json={"email": "carol@example.com", "password": "s3cret-pass!"})
    assert ok.status_code == 200 and ok.json()["user"]["email"] == "carol@example.com"
    bad = client.post("/v1/auth/login", json={"email": "carol@example.com", "password": "wrong-pass!"})
    assert bad.status_code == 401
    unknown = client.post("/v1/auth/login", json={"email": "nobody@example.com", "password": "whatever1"})
    assert unknown.status_code == 401


def test_refresh_flow(client):
    t = register(client)
    r = client.post("/v1/auth/refresh", json={"refresh_token": t["refresh_token"]})
    assert r.status_code == 200 and r.json()["access_token"]
    # an access token must not work as a refresh token, and vice versa
    assert client.post("/v1/auth/refresh", json={"refresh_token": t["access_token"]}).status_code == 401
    assert client.get("/v1/me", headers={"Authorization": f"Bearer {t['refresh_token']}"}).status_code == 401
    assert client.post("/v1/auth/refresh", json={"refresh_token": "garbage"}).status_code == 401


def test_protected_routes_need_token(client):
    assert client.get("/v1/me").status_code == 401
    assert client.get("/v1/tasks").status_code == 401
    assert client.get("/v1/me", headers={"Authorization": "Bearer nope"}).status_code == 401
    t = register(client)
    assert client.get("/v1/me", headers=auth_headers(t)).status_code == 200
