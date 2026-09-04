from app.services.storage import storage_path


def test_storage_path_keeps_private_object_key():
  assert storage_path("user-1/outfit-1.jpg") == "user-1/outfit-1.jpg"


def test_storage_path_migrates_legacy_public_url():
  value = "https://project.supabase.co/storage/v1/object/public/outfits/user-1/outfit-1.jpg"
  assert storage_path(value) == "user-1/outfit-1.jpg"


def test_storage_path_removes_signed_query_token():
  value = "https://project.supabase.co/storage/v1/object/sign/outfits/user-1/outfit-1.jpg?token=secret"
  assert storage_path(value) == "user-1/outfit-1.jpg"


def test_unrelated_external_url_is_never_signed_as_an_outfit_object():
  assert storage_path("https://example.com/photo.jpg") is None
