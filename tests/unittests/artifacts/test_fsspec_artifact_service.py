# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the read-only FsspecArtifactService.

The cloud backends are covered by hand-written fakes rather than mocks of
gcsfs/s3fs, because the behavior under test *is* the difference between the
two clients: gcsfs returns user metadata from listings, while s3fs omits it
from ``info()`` entirely and exposes it through a separate ``metadata()`` call.
The fakes encode exactly those semantics. A separate test drives the service
through a real ``fsspec`` memory filesystem so the generic path is exercised
against genuine fsspec rather than a fake.
"""

from __future__ import annotations

import datetime
from typing import Any
from typing import Optional

from google.adk.artifacts.base_artifact_service import ArtifactVersion
from google.adk.artifacts.fsspec_artifact_service import FsspecArtifactService
from google.adk.errors.input_validation_error import InputValidationError
from google.genai import types
import pytest

FIXED_DATETIME = datetime.datetime(
    2025, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
)

APP_NAME = "app0"
USER_ID = "user0"
SESSION_ID = "123"


class _FakeObject:
  """One stored object: its bytes plus whatever metadata the backend keeps."""

  def __init__(
      self,
      data: bytes,
      content_type: Optional[str],
      metadata: dict[str, str],
      create_time: datetime.datetime,
  ):
    self.data = data
    self.content_type = content_type
    self.metadata = metadata
    self.create_time = create_time


class _FakeFileSystem:
  """The subset of the fsspec filesystem API this service actually calls.

  Subclasses supply the per-backend shape of ``info`` and ``find`` results.
  """

  protocol: tuple[str, ...] = ()

  def __init__(self) -> None:
    self.objects: dict[str, _FakeObject] = {}
    self.metadata_calls: list[str] = []

  def put(
      self,
      key: str,
      data: bytes,
      *,
      content_type: Optional[str] = None,
      metadata: Optional[dict[str, str]] = None,
      create_time: datetime.datetime = FIXED_DATETIME,
  ) -> None:
    """Seeds an object, standing in for a write by another service."""
    self.objects[key] = _FakeObject(
        data=data,
        content_type=content_type,
        metadata=dict(metadata or {}),
        create_time=create_time,
    )

  def _strip_protocol(self, path: str) -> str:
    for protocol in self.protocol:
      prefix = f"{protocol}://"
      if path.startswith(prefix):
        return path[len(prefix) :].rstrip("/")
    return path.rstrip("/")

  def unstrip_protocol(self, name: str) -> str:
    return f"{self.protocol[0]}://{name}"

  def cat_file(self, key: str) -> bytes:
    if key not in self.objects:
      raise FileNotFoundError(key)
    return self.objects[key].data

  def _info(self, key: str) -> dict[str, Any]:
    raise NotImplementedError()

  def info(self, key: str) -> dict[str, Any]:
    if key not in self.objects:
      raise FileNotFoundError(key)
    return self._info(key)

  def find(self, path: str, detail: bool = False) -> Any:
    # Object stores have a flat namespace, so a "directory" listing is a
    # prefix match. Matches how gcsfs and s3fs behave for this layout.
    prefix = f"{path.rstrip('/')}/"
    keys = sorted(k for k in self.objects if k.startswith(prefix))
    if not detail:
      return keys
    return {key: self._info(key) for key in keys}


class _FakeGcsFileSystem(_FakeFileSystem):
  """Mimics gcsfs: user metadata rides along in every info dict.

  gcsfs copies the whole GCS object resource into its info dicts, so a single
  ``find(detail=True)`` carries the metadata of every version.
  """

  protocol = ("gs", "gcs")

  def _info(self, key: str) -> dict[str, Any]:
    obj = self.objects[key]
    return {
        "name": key,
        "size": len(obj.data),
        "type": "file",
        "metadata": dict(obj.metadata),
        "contentType": obj.content_type,
        "ctime": obj.create_time,
        "mtime": obj.create_time,
    }


class _FakeS3FileSystem(_FakeFileSystem):
  """Mimics s3fs: user metadata is absent from info and needs a head_object.

  s3fs's ``_info`` returns a fixed set of keys that excludes user metadata,
  and its ``metadata()`` rewrites "_" to "-" in metadata keys.
  """

  protocol = ("s3", "s3a")

  def _info(self, key: str) -> dict[str, Any]:
    obj = self.objects[key]
    return {
        "name": key,
        "size": len(obj.data),
        "type": "file",
        "ETag": "etag",
        "StorageClass": "STANDARD",
        "ContentType": obj.content_type,
        "LastModified": obj.create_time,
    }

  def metadata(self, key: str) -> dict[str, str]:
    self.metadata_calls.append(key)
    if key not in self.objects:
      raise FileNotFoundError(key)
    return {
        k.replace("_", "-"): v for k, v in self.objects[key].metadata.items()
    }


@pytest.fixture(params=["gcs", "s3"])
def backend(request):
  """Yields a fake cloud filesystem for each supported native backend."""
  if request.param == "gcs":
    return _FakeGcsFileSystem()
  return _FakeS3FileSystem()


def _service(fs: _FakeFileSystem, root: str = "test_bucket"):
  """Builds a service over a fake filesystem rooted at a bucket or folder."""
  return FsspecArtifactService(f"{fs.protocol[0]}://{root}", fs=fs)


def _session_key(filename: str, version: int, root: str = "test_bucket") -> str:
  return f"{root}/{APP_NAME}/{USER_ID}/{SESSION_ID}/{filename}/{version}"


def _user_key(filename: str, version: int, root: str = "test_bucket") -> str:
  return f"{root}/{APP_NAME}/{USER_ID}/user/{filename}/{version}"


@pytest.mark.asyncio
async def test_load_artifact_returns_binary_part(backend):
  """A plain binary artifact round-trips with its stored MIME type."""
  backend.put(
      _session_key("file456", 0), b"test_data", content_type="text/plain"
  )
  service = _service(backend)

  assert await service.load_artifact(
      app_name=APP_NAME,
      user_id=USER_ID,
      session_id=SESSION_ID,
      filename="file456",
  ) == types.Part.from_bytes(data=b"test_data", mime_type="text/plain")


@pytest.mark.asyncio
async def test_load_artifact_returns_text_part_when_flagged(backend):
  """The adkIsText flag reconstructs a text Part rather than inline_data.

  Text and bytes serialize identically, so without the flag a text artifact
  would come back as a non-equal inline_data Part.
  """
  backend.put(
      _session_key("notes", 0),
      b"hello world",
      content_type="text/plain",
      metadata={"adkIsText": "true"},
  )
  service = _service(backend)

  assert await service.load_artifact(
      app_name=APP_NAME,
      user_id=USER_ID,
      session_id=SESSION_ID,
      filename="notes",
  ) == types.Part(text="hello world")


@pytest.mark.asyncio
async def test_load_artifact_preserves_display_name(backend):
  """adkDisplayName is restored onto the returned Blob."""
  backend.put(
      _session_key("chart.png", 0),
      b"\x89PNG",
      content_type="image/png",
      metadata={"adkDisplayName": "Quarterly chart"},
  )
  service = _service(backend)

  part = await service.load_artifact(
      app_name=APP_NAME,
      user_id=USER_ID,
      session_id=SESSION_ID,
      filename="chart.png",
  )

  assert part == types.Part(
      inline_data=types.Blob(
          mime_type="image/png",
          data=b"\x89PNG",
          display_name="Quarterly chart",
      )
  )


@pytest.mark.asyncio
async def test_load_artifact_returns_file_data_uri(backend):
  """A file_data artifact stores an empty payload and a URI in metadata."""
  backend.put(
      _session_key("remote.pdf", 0),
      b"",
      content_type="application/pdf",
      metadata={
          "adkFileUri": "https://example.com/remote.pdf",
          "adkFileMimeType": "application/pdf",
      },
  )
  service = _service(backend)

  assert await service.load_artifact(
      app_name=APP_NAME,
      user_id=USER_ID,
      session_id=SESSION_ID,
      filename="remote.pdf",
  ) == types.Part(
      file_data=types.FileData(
          file_uri="https://example.com/remote.pdf",
          mime_type="application/pdf",
      )
  )


@pytest.mark.asyncio
async def test_load_artifact_reads_legacy_file_uri_key():
  """Objects written by older releases used a snake_case metadata key.

  GCS only: the legacy key was written by older GcsArtifactService releases,
  so objects carrying it exist nowhere else. It would not survive a read from
  S3 in any case, because s3fs rewrites "_" to "-" in metadata keys.
  """
  fs = _FakeGcsFileSystem()
  fs.put(
      _session_key("legacy.pdf", 0),
      b"",
      content_type="application/pdf",
      metadata={"file_uri": "https://example.com/legacy.pdf"},
  )
  service = _service(fs)

  part = await service.load_artifact(
      app_name=APP_NAME,
      user_id=USER_ID,
      session_id=SESSION_ID,
      filename="legacy.pdf",
  )

  assert part.file_data.file_uri == "https://example.com/legacy.pdf"


@pytest.mark.asyncio
async def test_load_artifact_resolves_artifact_reference(backend):
  """An artifact:// reference is followed to the artifact it names."""
  backend.put(
      _session_key("target.txt", 0),
      b"referenced payload",
      content_type="text/plain",
  )
  backend.put(
      _session_key("pointer.txt", 0),
      b"",
      metadata={
          "adkFileUri": (
              f"artifact://apps/{APP_NAME}/users/{USER_ID}/sessions/"
              f"{SESSION_ID}/artifacts/target.txt/versions/0"
          )
      },
  )
  service = _service(backend)

  assert await service.load_artifact(
      app_name=APP_NAME,
      user_id=USER_ID,
      session_id=SESSION_ID,
      filename="pointer.txt",
  ) == types.Part.from_bytes(
      data=b"referenced payload", mime_type="text/plain"
  )


@pytest.mark.asyncio
async def test_load_artifact_rejects_out_of_scope_reference(backend):
  """A reference may not escape the caller's app and user scope."""
  backend.put(
      _session_key("pointer.txt", 0),
      b"",
      metadata={
          "adkFileUri": (
              "artifact://apps/other_app/users/other_user/sessions/"
              "999/artifacts/secret.txt/versions/0"
          )
      },
  )
  service = _service(backend)

  with pytest.raises(InputValidationError, match="same app and user scope"):
    await service.load_artifact(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        filename="pointer.txt",
    )


@pytest.mark.asyncio
async def test_load_artifact_rejects_malformed_reference(backend):
  """A malformed artifact:// URI is reported rather than silently ignored."""
  backend.put(
      _session_key("pointer.txt", 0),
      b"",
      metadata={"adkFileUri": "artifact://nonsense"},
  )
  service = _service(backend)

  with pytest.raises(InputValidationError, match="Invalid artifact reference"):
    await service.load_artifact(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        filename="pointer.txt",
    )


@pytest.mark.asyncio
async def test_load_artifact_defaults_to_latest_version(backend):
  """Omitting the version loads the highest one stored."""
  for version in range(3):
    backend.put(
        _session_key("file456", version),
        f"v{version}".encode(),
        content_type="text/plain",
    )
  service = _service(backend)

  assert await service.load_artifact(
      app_name=APP_NAME,
      user_id=USER_ID,
      session_id=SESSION_ID,
      filename="file456",
  ) == types.Part.from_bytes(data=b"v2", mime_type="text/plain")


@pytest.mark.asyncio
async def test_load_artifact_specific_version(backend):
  """An explicit version loads exactly that version."""
  for version in range(3):
    backend.put(
        _session_key("file456", version),
        f"v{version}".encode(),
        content_type="text/plain",
    )
  service = _service(backend)

  assert await service.load_artifact(
      app_name=APP_NAME,
      user_id=USER_ID,
      session_id=SESSION_ID,
      filename="file456",
      version=1,
  ) == types.Part.from_bytes(data=b"v1", mime_type="text/plain")


@pytest.mark.asyncio
async def test_load_artifact_missing_returns_none(backend):
  """Loading an artifact that was never stored yields None, not an error."""
  service = _service(backend)

  assert (
      await service.load_artifact(
          app_name=APP_NAME,
          user_id=USER_ID,
          session_id=SESSION_ID,
          filename="absent",
      )
      is None
  )


@pytest.mark.asyncio
async def test_load_artifact_missing_version_returns_none(backend):
  """Requesting a version that does not exist yields None."""
  backend.put(_session_key("file456", 0), b"v0", content_type="text/plain")
  service = _service(backend)

  assert (
      await service.load_artifact(
          app_name=APP_NAME,
          user_id=USER_ID,
          session_id=SESSION_ID,
          filename="file456",
          version=7,
      )
      is None
  )


@pytest.mark.asyncio
async def test_load_artifact_from_user_namespace(backend):
  """A "user:" filename reads from the user-scoped location."""
  backend.put(
      _user_key("user:shared.txt", 0),
      b"shared",
      content_type="text/plain",
      metadata={"adkIsText": "true"},
  )
  service = _service(backend)

  assert await service.load_artifact(
      app_name=APP_NAME,
      user_id=USER_ID,
      session_id=SESSION_ID,
      filename="user:shared.txt",
  ) == types.Part(text="shared")


@pytest.mark.asyncio
async def test_load_artifact_user_namespace_without_session(backend):
  """User-scoped artifacts are readable with no session at all."""
  backend.put(
      _user_key("user:shared.txt", 0), b"shared", content_type="text/plain"
  )
  service = _service(backend)

  assert await service.load_artifact(
      app_name=APP_NAME, user_id=USER_ID, filename="user:shared.txt"
  ) == types.Part.from_bytes(data=b"shared", mime_type="text/plain")


@pytest.mark.asyncio
async def test_session_scoped_read_requires_session_id(backend):
  """A session-scoped filename cannot be resolved without a session."""
  service = _service(backend)

  with pytest.raises(InputValidationError, match="Session ID must be provided"):
    await service.load_artifact(
        app_name=APP_NAME, user_id=USER_ID, filename="file456"
    )


@pytest.mark.asyncio
async def test_list_versions(backend):
  """Versions come back sorted ascending."""
  for version in (2, 0, 1):
    backend.put(_session_key("file456", version), b"x")
  service = _service(backend)

  assert await service.list_versions(
      app_name=APP_NAME,
      user_id=USER_ID,
      session_id=SESSION_ID,
      filename="file456",
  ) == [0, 1, 2]


@pytest.mark.asyncio
async def test_list_versions_excludes_nested_artifact(backend):
  """A nested artifact's versions do not leak into its parent's.

  Filenames may contain "/", so "doc" is a path prefix of "doc/nested". Only
  objects whose remaining segment is a bare number are versions of "doc".
  """
  backend.put(_session_key("doc", 0), b"parent")
  backend.put(_session_key("doc/nested", 5), b"child")
  service = _service(backend)

  assert await service.list_versions(
      app_name=APP_NAME,
      user_id=USER_ID,
      session_id=SESSION_ID,
      filename="doc",
  ) == [0]
  assert await service.list_versions(
      app_name=APP_NAME,
      user_id=USER_ID,
      session_id=SESSION_ID,
      filename="doc/nested",
  ) == [5]


@pytest.mark.asyncio
async def test_list_versions_of_missing_artifact_is_empty(backend):
  """An artifact with no stored objects has no versions."""
  service = _service(backend)

  assert (
      await service.list_versions(
          app_name=APP_NAME,
          user_id=USER_ID,
          session_id=SESSION_ID,
          filename="absent",
      )
      == []
  )


@pytest.mark.asyncio
async def test_list_artifact_keys_covers_session_and_user_scopes(backend):
  """Listing a session returns its own artifacts plus user-scoped ones."""
  backend.put(_session_key("session_file", 0), b"x")
  backend.put(_user_key("user:shared", 0), b"x")
  service = _service(backend)

  assert await service.list_artifact_keys(
      app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
  ) == ["session_file", "user:shared"]


@pytest.mark.asyncio
async def test_list_artifact_keys_without_session_is_user_scoped(backend):
  """Omitting the session lists only the user-scoped artifacts."""
  backend.put(_session_key("session_file", 0), b"x")
  backend.put(_user_key("user:shared", 0), b"x")
  service = _service(backend)

  assert await service.list_artifact_keys(
      app_name=APP_NAME, user_id=USER_ID
  ) == ["user:shared"]


@pytest.mark.asyncio
async def test_list_artifact_keys_includes_nested_filenames(backend):
  """Filenames containing "/" are reported whole, not truncated."""
  backend.put(_session_key("images/photo.png", 0), b"x")
  backend.put(_session_key("images/photo.png", 1), b"x")
  service = _service(backend)

  assert await service.list_artifact_keys(
      app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
  ) == ["images/photo.png"]


@pytest.mark.asyncio
async def test_list_artifact_keys_isolates_other_users(backend):
  """One user's artifacts never appear in another user's listing."""
  backend.put(_session_key("mine", 0), b"x")
  backend.put(f"test_bucket/{APP_NAME}/other_user/{SESSION_ID}/theirs/0", b"x")
  service = _service(backend)

  assert await service.list_artifact_keys(
      app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
  ) == ["mine"]


@pytest.mark.asyncio
async def test_list_artifact_versions_reports_metadata(backend):
  """Every version's metadata is returned, sorted by version."""
  for version in range(3):
    backend.put(
        _session_key("file456", version),
        f"v{version}".encode(),
        content_type="text/plain",
        metadata={"key": f"value{version}"},
    )
  service = _service(backend)

  artifact_versions = await service.list_artifact_versions(
      app_name=APP_NAME,
      user_id=USER_ID,
      session_id=SESSION_ID,
      filename="file456",
  )

  assert artifact_versions == [
      ArtifactVersion(
          version=version,
          canonical_uri=(
              f"{backend.protocol[0]}://{_session_key('file456', version)}"
          ),
          custom_metadata={"key": f"value{version}"},
          mime_type="text/plain",
          create_time=FIXED_DATETIME.timestamp(),
      )
      for version in range(3)
  ]


@pytest.mark.asyncio
async def test_list_artifact_versions_excludes_nested_artifact(backend):
  """Version metadata listings apply the same nesting rule as list_versions."""
  backend.put(_session_key("doc", 0), b"parent", content_type="text/plain")
  backend.put(_session_key("doc/nested", 5), b"child", content_type="text/plain")
  service = _service(backend)

  artifact_versions = await service.list_artifact_versions(
      app_name=APP_NAME,
      user_id=USER_ID,
      session_id=SESSION_ID,
      filename="doc",
  )

  assert [av.version for av in artifact_versions] == [0]


@pytest.mark.asyncio
async def test_get_artifact_version_defaults_to_latest(backend):
  """Omitting the version returns metadata for the highest one."""
  for version in range(3):
    backend.put(
        _session_key("file456", version), b"x", content_type="text/plain"
    )
  service = _service(backend)

  artifact_version = await service.get_artifact_version(
      app_name=APP_NAME,
      user_id=USER_ID,
      session_id=SESSION_ID,
      filename="file456",
  )

  assert artifact_version.version == 2
  assert artifact_version.canonical_uri == (
      f"{backend.protocol[0]}://{_session_key('file456', 2)}"
  )


@pytest.mark.asyncio
async def test_get_artifact_version_specific(backend):
  """An explicit version returns that version's metadata."""
  for version in range(3):
    backend.put(
        _session_key("file456", version), b"x", content_type="text/plain"
    )
  service = _service(backend)

  artifact_version = await service.get_artifact_version(
      app_name=APP_NAME,
      user_id=USER_ID,
      session_id=SESSION_ID,
      filename="file456",
      version=1,
  )

  assert artifact_version.version == 1


@pytest.mark.asyncio
async def test_get_artifact_version_missing_returns_none(backend):
  """Metadata for an artifact that does not exist is None."""
  service = _service(backend)

  assert (
      await service.get_artifact_version(
          app_name=APP_NAME,
          user_id=USER_ID,
          session_id=SESSION_ID,
          filename="absent",
      )
      is None
  )


@pytest.mark.asyncio
async def test_get_artifact_version_out_of_range_returns_none(backend):
  """Metadata for a version beyond those stored is None."""
  backend.put(_session_key("file456", 0), b"x")
  service = _service(backend)

  assert (
      await service.get_artifact_version(
          app_name=APP_NAME,
          user_id=USER_ID,
          session_id=SESSION_ID,
          filename="file456",
          version=9,
      )
      is None
  )


@pytest.mark.asyncio
async def test_root_may_be_a_folder_within_a_bucket(backend):
  """A URI pointing at a folder scopes every key beneath it."""
  root = "test_bucket/artifacts"
  backend.put(
      _session_key("file456", 0, root=root),
      b"nested_root",
      content_type="text/plain",
  )
  service = _service(backend, root=root)

  assert await service.load_artifact(
      app_name=APP_NAME,
      user_id=USER_ID,
      session_id=SESSION_ID,
      filename="file456",
  ) == types.Part.from_bytes(data=b"nested_root", mime_type="text/plain")

  artifact_version = await service.get_artifact_version(
      app_name=APP_NAME,
      user_id=USER_ID,
      session_id=SESSION_ID,
      filename="file456",
  )
  assert artifact_version.canonical_uri == (
      f"{backend.protocol[0]}://{root}/{APP_NAME}/{USER_ID}/{SESSION_ID}"
      "/file456/0"
  )


@pytest.mark.asyncio
async def test_canonical_uri_matches_gcs_artifact_service_format():
  """The GCS canonical URI is identical to the one GcsArtifactService emits.

  GcsArtifactService builds it as f"gs://{bucket_name}/{blob.name}"; this
  pins the fsspec service to the same string so the two agree on a bucket.
  """
  fs = _FakeGcsFileSystem()
  fs.put(_session_key("file456", 0), b"x", content_type="text/plain")
  service = _service(fs)

  artifact_version = await service.get_artifact_version(
      app_name=APP_NAME,
      user_id=USER_ID,
      session_id=SESSION_ID,
      filename="file456",
  )

  assert artifact_version.canonical_uri == (
      "gs://test_bucket/app0/user0/123/file456/0"
  )


@pytest.mark.asyncio
async def test_s3_reads_metadata_out_of_band():
  """On S3, user metadata needs a head_object because listings omit it."""
  fs = _FakeS3FileSystem()
  fs.put(
      _session_key("notes", 0),
      b"hello",
      content_type="text/plain",
      metadata={"adkIsText": "true"},
  )
  service = _service(fs)

  assert await service.load_artifact(
      app_name=APP_NAME,
      user_id=USER_ID,
      session_id=SESSION_ID,
      filename="notes",
  ) == types.Part(text="hello")
  assert fs.metadata_calls == [_session_key("notes", 0)]


@pytest.mark.asyncio
async def test_gcs_reads_all_version_metadata_in_one_listing():
  """On GCS a versions listing costs one call, with no per-object lookups."""
  fs = _FakeGcsFileSystem()
  calls: list[str] = []
  original_info = fs.info

  def counting_info(key: str):
    calls.append(key)
    return original_info(key)

  fs.info = counting_info
  for version in range(4):
    fs.put(
        _session_key("file456", version), b"x", content_type="text/plain"
    )
  service = _service(fs)

  artifact_versions = await service.list_artifact_versions(
      app_name=APP_NAME,
      user_id=USER_ID,
      session_id=SESSION_ID,
      filename="file456",
  )

  assert len(artifact_versions) == 4
  assert calls == []


@pytest.mark.asyncio
async def test_s3_reads_version_metadata_per_object():
  """On S3 a versions listing costs one head_object per version.

  S3's ListObjectsV2 never returns user metadata, so unlike GCS the metadata
  cannot be gathered from the listing alone. This pins that cost so a change
  in it is deliberate.
  """
  fs = _FakeS3FileSystem()
  for version in range(4):
    fs.put(_session_key("file456", version), b"x", content_type="text/plain")
  service = _service(fs)

  artifact_versions = await service.list_artifact_versions(
      app_name=APP_NAME,
      user_id=USER_ID,
      session_id=SESSION_ID,
      filename="file456",
  )

  assert len(artifact_versions) == 4
  assert fs.metadata_calls == [
      _session_key("file456", version) for version in range(4)
  ]


@pytest.mark.asyncio
async def test_writes_are_not_supported(backend):
  """Write paths fail loudly while the service is read-only."""
  service = _service(backend)

  with pytest.raises(NotImplementedError, match="read-only"):
    await service.save_artifact(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        filename="file456",
        artifact=types.Part(text="nope"),
    )

  with pytest.raises(NotImplementedError, match="read-only"):
    await service.delete_artifact(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        filename="file456",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_value",
    ["", "..", "a/../../b", "/absolute", "C:drive", "null\x00byte"],
)
async def test_path_segments_are_validated(backend, invalid_value):
  """Identifiers that could redirect the constructed path are rejected."""
  service = _service(backend)

  with pytest.raises(InputValidationError):
    await service.load_artifact(
        app_name=APP_NAME,
        user_id=invalid_value,
        session_id=SESSION_ID,
        filename="file456",
    )


@pytest.mark.asyncio
async def test_reads_through_a_real_fsspec_filesystem():
  """The generic path works against genuine fsspec, not just the fakes.

  A memory filesystem has no user-metadata concept, so custom metadata comes
  back empty and the MIME type is inferred from the filename.
  """
  fsspec = pytest.importorskip("fsspec")
  fs = fsspec.filesystem("memory", skip_instance_cache=True)
  fs.pipe_file(
      f"/test_bucket/{APP_NAME}/{USER_ID}/{SESSION_ID}/report.txt/0",
      b"plain bytes",
  )

  service = FsspecArtifactService("memory://test_bucket", fs=fs)

  assert await service.load_artifact(
      app_name=APP_NAME,
      user_id=USER_ID,
      session_id=SESSION_ID,
      filename="report.txt",
  ) == types.Part.from_bytes(data=b"plain bytes", mime_type="text/plain")

  assert await service.list_versions(
      app_name=APP_NAME,
      user_id=USER_ID,
      session_id=SESSION_ID,
      filename="report.txt",
  ) == [0]

  assert await service.list_artifact_keys(
      app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
  ) == ["report.txt"]

  artifact_version = await service.get_artifact_version(
      app_name=APP_NAME,
      user_id=USER_ID,
      session_id=SESSION_ID,
      filename="report.txt",
  )
  assert artifact_version.version == 0
  assert artifact_version.custom_metadata == {}
  # Reported metadata agrees with what load_artifact puts on the Part, even
  # though the backend itself supplies no content type.
  assert artifact_version.mime_type == "text/plain"


@pytest.mark.asyncio
async def test_unknown_backend_infers_mime_type_from_filename():
  """Backends without content types fall back to guessing from the name."""
  fsspec = pytest.importorskip("fsspec")
  fs = fsspec.filesystem("memory", skip_instance_cache=True)
  fs.pipe_file(
      f"/test_bucket/{APP_NAME}/{USER_ID}/{SESSION_ID}/diagram.png/0",
      b"\x89PNG",
  )

  service = FsspecArtifactService("memory://test_bucket", fs=fs)

  part = await service.load_artifact(
      app_name=APP_NAME,
      user_id=USER_ID,
      session_id=SESSION_ID,
      filename="diagram.png",
  )

  assert part.inline_data.mime_type == "image/png"
