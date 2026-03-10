from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_ok():
  resp = client.get("/health")
  assert resp.status_code == 200
  assert resp.json() == {"status": "ok"}


def test_score_requires_image_and_context():
  # Missing required multipart image -> FastAPI validation should reject with 422
  resp = client.post("/v1/outfits/score")
  assert resp.status_code == 422
