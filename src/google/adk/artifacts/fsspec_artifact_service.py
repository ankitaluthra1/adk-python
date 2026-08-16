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

"""A read-only artifact service backed by fsspec.

``fsspec`` exposes one filesystem API over many storage backends, so a single
artifact service implementation reaches Google Cloud Storage (``gcsfs``),
Amazon S3 (``s3fs``), and anything else with an fsspec driver.

The object layout matches :class:`~google.adk.artifacts.gcs_artifact_service.GcsArtifactService`
exactly, so this service reads buckets that service already wrote:
  - For files with user namespace (starting with "user:"):
    {root}/{app_name}/{user_id}/user/{filename}/{version}
  - For regular session-scoped files:
    {root}/{app_name}/{user_id}/{session_id}/{filename}/{version}

``root`` is whatever the configured URI points at, so both a bucket
(``gs://bucket``) and a folder within one (``gs://bucket/artifacts``) work.

This is the first step of a larger change and is **read only**:
``save_artifact`` and ``delete_artifact`` raise ``NotImplementedError``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import logging
import mimetypes
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

from google.genai import types
from typing_extensions import override

from . import artifact_util
from ..errors.input_validation_error import InputValidationError
from .base_artifact_service import ArtifactVersion
from .base_artifact_service import BaseArtifactService

if TYPE_CHECKING:
  from fsspec import AbstractFileSystem

logger = logging.getLogger("google_adk." + __name__)

# TODO: These four keys are copied verbatim from gcs_artifact_service.py so
# that both services read and write the same objects. Once this service also
# supports writes, hoist them into a module both can import.
_DISPLAY_NAME_METADATA_KEY = "adkDisplayName"
_IS_TEXT_METADATA_KEY = "adkIsText"
_FILE_URI_METADATA_KEY = "adkFileUri"
_FILE_MIME_TYPE_METADATA_KEY = "adkFileMimeType"

# Written by older releases of GcsArtifactService.
_LEGACY_FILE_URI_METADATA_KEY = "file_uri"

_DEFAULT_MIME_TYPE = "application/octet-stream"

_GCS_PROTOCOLS = frozenset({"gs", "gcs"})
_S3_PROTOCOLS = frozenset({"s3", "s3a"})


@dataclasses.dataclass
class _ObjectMetadata:
  """Storage metadata for a single stored artifact version.

  Attributes:
    custom_metadata: User-defined key/value pairs attached to the object.
    content_type: The object's declared MIME type, if the backend reports one.
    create_time: Unix timestamp (seconds) the object was created, if known.
  """

  custom_metadata: dict[str, Any]
  content_type: Optional[str]
  create_time: Optional[float]


def _to_timestamp(value: Any) -> Optional[float]:
  """Normalizes a backend-supplied time to a Unix timestamp.

  Backends report times inconsistently: gcsfs and s3fs hand back
  ``datetime`` objects, while the local and memory filesystems use floats.

  Args:
    value: The raw value taken from an fsspec info dict.

  Returns:
    The time in seconds since the epoch, or None if it cannot be interpreted.
  """
  if isinstance(value, datetime.datetime):
    return value.timestamp()
  if isinstance(value, (int, float)):
    return float(value)
  return None


class _MetadataReader:
  """Reads storage metadata for one fsspec backend.

  fsspec has no portable API for user-defined object metadata: ``pipe_file``
  takes no content type and ``AbstractFileSystem`` has no metadata concept at
  all. Each backend that does support it spells it differently, so the details
  live in a subclass per backend.
  """

  def read(
      self, fs: AbstractFileSystem, key: str
  ) -> Optional[_ObjectMetadata]:
    """Reads metadata for a single object.

    Args:
      fs: The filesystem holding the object.
      key: Full path of the object within that filesystem.

    Returns:
      The object's metadata, or None if the object does not exist.
    """
    raise NotImplementedError()

  def read_many(
      self, fs: AbstractFileSystem, prefix: str
  ) -> dict[str, _ObjectMetadata]:
    """Reads metadata for every object beneath a prefix.

    Args:
      fs: The filesystem to scan.
      prefix: Path prefix to list, without a trailing separator.

    Returns:
      A mapping of object path to metadata. Empty if the prefix holds nothing.
    """
    raise NotImplementedError()


def _generic_metadata(info: dict[str, Any]) -> _ObjectMetadata:
  """Builds metadata from the keys any fsspec backend may supply."""
  return _ObjectMetadata(
      custom_metadata={},
      content_type=info.get("contentType") or info.get("ContentType"),
      create_time=_to_timestamp(
          info.get("created") or info.get("ctime") or info.get("mtime")
      ),
  )


class _GenericMetadataReader(_MetadataReader):
  """Best-effort reader for backends without user-defined metadata.

  Local disk, in-memory and SFTP filesystems store bytes and nothing else, so
  custom metadata always comes back empty and the MIME type is left to the
  caller to infer from the filename.
  """

  @override
  def read(
      self, fs: AbstractFileSystem, key: str
  ) -> Optional[_ObjectMetadata]:
    try:
      info = fs.info(key)
    except FileNotFoundError:
      return None
    return _generic_metadata(info)

  @override
  def read_many(
      self, fs: AbstractFileSystem, prefix: str
  ) -> dict[str, _ObjectMetadata]:
    return {
        key: _generic_metadata(info)
        for key, info in _find_detail(fs, prefix).items()
    }


def _gcs_metadata(info: dict[str, Any]) -> _ObjectMetadata:
  """Builds metadata from a gcsfs info dict.

  gcsfs copies the whole GCS object resource into its info dicts, so the
  user-defined ``metadata`` map and ``contentType`` are present on both
  ``info()`` and ``find(detail=True)`` results.

  Args:
    info: An info dict produced by gcsfs.

  Returns:
    The corresponding object metadata.
  """
  return _ObjectMetadata(
      custom_metadata=dict(info.get("metadata") or {}),
      content_type=info.get("contentType"),
      create_time=_to_timestamp(info.get("ctime")),
  )


class _GcsMetadataReader(_MetadataReader):
  """Reads object metadata from Google Cloud Storage via gcsfs."""

  @override
  def read(
      self, fs: AbstractFileSystem, key: str
  ) -> Optional[_ObjectMetadata]:
    try:
      info = fs.info(key)
    except FileNotFoundError:
      return None
    return _gcs_metadata(info)

  @override
  def read_many(
      self, fs: AbstractFileSystem, prefix: str
  ) -> dict[str, _ObjectMetadata]:
    # A single listing carries every version's metadata, so this costs one
    # request no matter how many versions exist.
    return {
        key: _gcs_metadata(info)
        for key, info in _find_detail(fs, prefix).items()
    }


def _s3_custom_metadata(fs: AbstractFileSystem, key: str) -> dict[str, Any]:
  """Fetches user-defined metadata for one S3 object.

  S3's ``ListObjectsV2`` never returns user metadata, so this costs a
  ``head_object`` call per object.

  Args:
    fs: The s3fs filesystem.
    key: Full path of the object.

  Returns:
    The object's user metadata, empty if it has none or cannot be read.
  """
  try:
    return dict(fs.metadata(key) or {})
  except FileNotFoundError:
    return {}


class _S3MetadataReader(_MetadataReader):
  """Reads object metadata from Amazon S3 via s3fs.

  s3fs deliberately omits user metadata from ``info()``, exposing it through
  ``metadata()`` instead, and reports only ``LastModified`` rather than a
  creation time. Note that s3fs rewrites ``_`` to ``-`` in metadata keys, so
  a custom key ``my_key`` reads back as ``my-key``.
  """

  def _metadata(
      self, fs: AbstractFileSystem, key: str, info: dict[str, Any]
  ) -> _ObjectMetadata:
    return _ObjectMetadata(
        custom_metadata=_s3_custom_metadata(fs, key),
        content_type=info.get("ContentType"),
        # S3 objects are immutable, so last-modified is also their create time.
        create_time=_to_timestamp(info.get("LastModified")),
    )

  @override
  def read(
      self, fs: AbstractFileSystem, key: str
  ) -> Optional[_ObjectMetadata]:
    try:
      info = fs.info(key)
    except FileNotFoundError:
      return None
    return self._metadata(fs, key, info)

  @override
  def read_many(
      self, fs: AbstractFileSystem, prefix: str
  ) -> dict[str, _ObjectMetadata]:
    return {
        key: self._metadata(fs, key, info)
        for key, info in _find_detail(fs, prefix).items()
    }


def _select_metadata_reader(fs: AbstractFileSystem) -> _MetadataReader:
  """Picks the metadata reader matching a filesystem's backend.

  Args:
    fs: The filesystem to inspect.

  Returns:
    A reader for that backend, falling back to best-effort behavior for
    backends that store no user-defined metadata.
  """
  protocol = fs.protocol
  protocols = set(protocol) if isinstance(protocol, tuple) else {protocol}
  if protocols & _GCS_PROTOCOLS:
    return _GcsMetadataReader()
  if protocols & _S3_PROTOCOLS:
    return _S3MetadataReader()
  logger.info(
      "Filesystem protocol %s stores no user-defined object metadata; custom"
      " metadata will be empty and MIME types inferred from filenames.",
      protocol,
  )
  return _GenericMetadataReader()


def _find_detail(
    fs: AbstractFileSystem, prefix: str
) -> dict[str, dict[str, Any]]:
  """Lists every object beneath a prefix with its info dict.

  Args:
    fs: The filesystem to scan.
    prefix: Path prefix to list.

  Returns:
    A mapping of object path to info dict, empty if the prefix holds nothing.
  """
  try:
    found = fs.find(prefix, detail=True)
  except FileNotFoundError:
    # Most backends return an empty result for a missing prefix, but not all.
    return {}
  return {key: info for key, info in found.items() if info.get("type") != "directory"}


def _parse_version(key: str, prefix: str) -> Optional[int]:
  """Extracts the version of an artifact from one of its object paths.

  Object stores have a flat namespace, so listing by prefix is a plain string
  match with no notion of nesting depth. Because filenames are allowed to
  contain "/", the prefix of an artifact is also a prefix of every artifact
  nested under it: scanning "a/" to find versions of "a" also returns "a/b/3",
  which is version 3 of the distinct artifact "a/b".

  An object only holds a version of the artifact denoted by ``prefix`` when its
  path is exactly ``{prefix}{version}``, so anything with a further "/" in it
  belongs to some other artifact and must be skipped.

  TODO: This duplicates gcs_artifact_service._parse_version. Both should move
  to artifact_util once this service is no longer read-only.

  Args:
    key: The full path of the object, which must start with ``prefix``.
    prefix: The path prefix of the artifact, including the trailing "/".

  Returns:
    The version number, or None if the object does not hold a version of this
    artifact.
  """
  suffix = key[len(prefix) :]
  if "/" in suffix:
    # Belongs to a distinct artifact nested under this one.
    return None
  # int() also accepts surrounding whitespace, underscores and non-ASCII
  # digits, none of which the layout can produce.
  if not (suffix.isascii() and suffix.isdigit()):
    logger.warning(
        "Skipping object %s because it does not end with a version number.",
        key,
    )
    return None
  return int(suffix)


class FsspecArtifactService(BaseArtifactService):
  """A read-only artifact service backed by any fsspec filesystem.

  Reads artifacts written by
  :class:`~google.adk.artifacts.gcs_artifact_service.GcsArtifactService`, and
  extends the same layout to every other fsspec backend.

  Example:
    ```python
    from google.adk.artifacts.fsspec_artifact_service import (
        FsspecArtifactService,
    )

    # Requires `pip install gcsfs`.
    service = FsspecArtifactService("gs://my-bucket")

    # Requires `pip install s3fs`.
    service = FsspecArtifactService("s3://my-bucket/artifacts")
    ```

  Writes are not supported yet: ``save_artifact`` and ``delete_artifact``
  raise ``NotImplementedError``.
  """

  def __init__(
      self,
      path: str,
      *,
      fs: Optional[AbstractFileSystem] = None,
      **storage_options: Any,
  ):
    """Initializes the FsspecArtifactService.

    Args:
      path: URI of the bucket or folder holding the artifacts, for example
        ``gs://my-bucket`` or ``s3://my-bucket/artifacts``.
      fs: An already-configured fsspec filesystem to use instead of building
        one from ``path``. ``path`` still determines the root within it.
      **storage_options: Backend-specific options forwarded to fsspec, such as
        credentials. Ignored when ``fs`` is supplied.

    Raises:
      ImportError: If fsspec, or the driver for this URI's protocol, is not
        installed.
    """
    try:
      import fsspec  # pylint: disable=g-import-not-at-top
    except ImportError as exc:
      raise ImportError(
          "The 'fsspec' package is required to use FsspecArtifactService. "
          "Please install it by running: pip install fsspec"
      ) from exc

    if fs is None:
      # Listing caches make a write by another process invisible until they
      # expire. GcsArtifactService caches nothing, so match it.
      storage_options.setdefault("use_listings_cache", False)
      fs, root = fsspec.url_to_fs(path, **storage_options)
    else:
      root = fs._strip_protocol(path)  # pylint: disable=protected-access

    self._fs = fs
    self._root = root.rstrip("/")
    self._metadata_reader = _select_metadata_reader(fs)

  # TODO: Duplicated from GcsArtifactService._file_has_user_namespace; share
  # once this service is no longer read-only.
  def _file_has_user_namespace(self, filename: str) -> bool:
    """Checks if the filename has a user namespace.

    Args:
      filename: The filename to check.

    Returns:
      True if the filename has a user namespace (starts with "user:"),
      False otherwise.
    """
    return filename.startswith("user:")

  def _get_scope_prefix(
      self, app_name: str, user_id: str, session_id: Optional[str]
  ) -> str:
    """Builds the path prefix shared by every artifact in one scope."""
    artifact_util.validate_path_segment(app_name, "app_name")
    artifact_util.validate_path_segment(user_id, "user_id")
    if session_id is None:
      raise InputValidationError(
          "Session ID must be provided for session-scoped artifacts."
      )
    artifact_util.validate_path_segment(session_id, "session_id")
    return f"{self._root}/{app_name}/{user_id}/{session_id}"

  def _get_user_scope_prefix(self, app_name: str, user_id: str) -> str:
    """Builds the path prefix holding a user's namespaced artifacts."""
    artifact_util.validate_path_segment(app_name, "app_name")
    artifact_util.validate_path_segment(user_id, "user_id")
    return f"{self._root}/{app_name}/{user_id}/user"

  def _get_artifact_prefix(
      self,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
  ) -> str:
    """Builds the path prefix holding every version of one artifact."""
    if self._file_has_user_namespace(filename):
      scope = self._get_user_scope_prefix(app_name, user_id)
    else:
      scope = self._get_scope_prefix(app_name, user_id, session_id)
    return f"{scope}/{filename}"

  def _get_object_key(
      self,
      app_name: str,
      user_id: str,
      filename: str,
      version: int,
      session_id: Optional[str] = None,
  ) -> str:
    """Builds the full path of one stored artifact version."""
    prefix = self._get_artifact_prefix(
        app_name, user_id, filename, session_id
    )
    return f"{prefix}/{version}"

  def _resolve_mime_type(
      self, metadata: _ObjectMetadata, filename: str
  ) -> str:
    """Determines the MIME type to report for an artifact.

    Backends without a content-type concept report nothing, so the filename is
    used as a fallback.

    Args:
      metadata: Metadata read back from storage.
      filename: The artifact's filename, used to guess a type.

    Returns:
      The MIME type, defaulting to ``application/octet-stream``.
    """
    if metadata.content_type:
      return metadata.content_type
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or _DEFAULT_MIME_TYPE

  @override
  async def save_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      artifact: Union[types.Part, dict[str, Any]],
      session_id: Optional[str] = None,
      custom_metadata: Optional[dict[str, Any]] = None,
  ) -> int:
    raise NotImplementedError(
        "FsspecArtifactService is read-only; it cannot save artifacts yet."
    )

  @override
  async def delete_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
  ) -> None:
    raise NotImplementedError(
        "FsspecArtifactService is read-only; it cannot delete artifacts yet."
    )

  @override
  async def load_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
      version: Optional[int] = None,
  ) -> Optional[types.Part]:
    return await asyncio.to_thread(
        self._load_artifact,
        app_name,
        user_id,
        session_id,
        filename,
        version,
    )

  @override
  async def list_artifact_keys(
      self, *, app_name: str, user_id: str, session_id: Optional[str] = None
  ) -> list[str]:
    return await asyncio.to_thread(
        self._list_artifact_keys,
        app_name,
        user_id,
        session_id,
    )

  @override
  async def list_versions(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
  ) -> list[int]:
    return await asyncio.to_thread(
        self._list_versions,
        app_name,
        user_id,
        session_id,
        filename,
    )

  @override
  async def list_artifact_versions(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
  ) -> list[ArtifactVersion]:
    return await asyncio.to_thread(
        self._list_artifact_versions,
        app_name,
        user_id,
        session_id,
        filename,
    )

  @override
  async def get_artifact_version(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
      version: Optional[int] = None,
  ) -> Optional[ArtifactVersion]:
    return await asyncio.to_thread(
        self._get_artifact_version,
        app_name,
        user_id,
        session_id,
        filename,
        version,
    )

  def _load_artifact(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      filename: str,
      version: Optional[int] = None,
  ) -> Optional[types.Part]:
    """Loads one version of an artifact from storage."""
    if version is None:
      versions = self._list_versions(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=filename,
      )
      if not versions:
        return None
      version = max(versions)

    key = self._get_object_key(
        app_name, user_id, filename, version, session_id
    )
    metadata = self._metadata_reader.read(self._fs, key)
    if metadata is None:
      return None

    file_uri = metadata.custom_metadata.get(
        _FILE_URI_METADATA_KEY
    ) or metadata.custom_metadata.get(_LEGACY_FILE_URI_METADATA_KEY)
    if file_uri:
      return self._load_file_data_artifact(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          file_uri=file_uri,
          metadata=metadata,
      )

    try:
      data = self._fs.cat_file(key)
    except FileNotFoundError:
      # Deleted between the metadata read and the payload read.
      logger.warning("Artifact payload %s disappeared while loading.", key)
      return None

    if metadata.custom_metadata.get(_IS_TEXT_METADATA_KEY) == "true":
      return types.Part(text=data.decode("utf-8"))

    mime_type = self._resolve_mime_type(metadata, filename)
    display_name = metadata.custom_metadata.get(_DISPLAY_NAME_METADATA_KEY)
    if display_name:
      return types.Part(
          inline_data=types.Blob(
              mime_type=mime_type,
              data=data,
              display_name=display_name,
          )
      )
    return types.Part.from_bytes(data=data, mime_type=mime_type)

  def _load_file_data_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      file_uri: str,
      metadata: _ObjectMetadata,
  ) -> Optional[types.Part]:
    """Rebuilds an artifact whose payload is a URI rather than bytes.

    Args:
      app_name: The name of the application.
      user_id: The ID of the user.
      session_id: The ID of the session, or None for user-scoped artifacts.
      file_uri: The stored URI.
      metadata: Metadata read back for the stored object.

    Returns:
      The artifact, or None if an ``artifact://`` reference cannot be resolved.

    Raises:
      InputValidationError: If the reference URI is malformed or points
        outside the caller's scope.
    """
    if file_uri.startswith("artifact://"):
      parsed_uri = artifact_util.parse_artifact_uri(file_uri)
      if not parsed_uri:
        raise InputValidationError(f"Invalid artifact reference URI: {file_uri}")
      artifact_util.validate_artifact_reference_scope(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          parsed_uri=parsed_uri,
      )
      return self._load_artifact(
          app_name=parsed_uri.app_name,
          user_id=parsed_uri.user_id,
          session_id=parsed_uri.session_id,
          filename=parsed_uri.filename,
          version=parsed_uri.version,
      )

    mime_type = (
        metadata.custom_metadata.get(_FILE_MIME_TYPE_METADATA_KEY)
        or metadata.content_type
    )
    return types.Part(
        file_data=types.FileData(file_uri=file_uri, mime_type=mime_type)
    )

  def _list_artifact_keys(
      self, app_name: str, user_id: str, session_id: Optional[str]
  ) -> list[str]:
    """Lists the filenames visible to a session and its user."""
    artifact_util.validate_path_segment(app_name, "app_name")
    artifact_util.validate_path_segment(user_id, "user_id")
    filenames: set[str] = set()

    if session_id is not None:
      session_prefix = self._get_scope_prefix(app_name, user_id, session_id)
      filenames.update(self._filenames_under(session_prefix))

    user_prefix = self._get_user_scope_prefix(app_name, user_id)
    filenames.update(self._filenames_under(user_prefix))

    return sorted(filenames)

  def _filenames_under(self, scope_prefix: str) -> set[str]:
    """Extracts artifact filenames from every object in one scope.

    Args:
      scope_prefix: Path prefix of the scope, without a trailing separator.

    Returns:
      The filenames found, with the version segment stripped. Filenames may
      themselves contain "/", so only the final segment is removed.
    """
    prefix = f"{scope_prefix}/"
    filenames: set[str] = set()
    for key in _find_detail(self._fs, scope_prefix):
      relative = key[len(prefix) :]
      filename = "/".join(relative.split("/")[:-1])
      if filename:
        filenames.add(filename)
    return filenames

  def _list_versions(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      filename: str,
  ) -> list[int]:
    """Lists every stored version of one artifact, in ascending order."""
    artifact_prefix = self._get_artifact_prefix(
        app_name, user_id, filename, session_id
    )
    prefix = f"{artifact_prefix}/"
    versions = []
    for key in _find_detail(self._fs, artifact_prefix):
      version = _parse_version(key, prefix)
      if version is None:
        continue
      versions.append(version)
    versions.sort()
    return versions

  def _build_artifact_version(
      self,
      key: str,
      version: int,
      filename: str,
      metadata: _ObjectMetadata,
  ) -> ArtifactVersion:
    """Assembles the metadata record describing one stored version."""
    fields: dict[str, Any] = {
        "version": version,
        # Restores the protocol fsspec strips from paths, so GCS objects get a
        # gs:// URI, S3 objects an s3:// one, and so on.
        "canonical_uri": self._fs.unstrip_protocol(key),
        "custom_metadata": metadata.custom_metadata,
        # Resolved the same way as in _load_artifact, so the type reported
        # here always matches the type on the Part that load returns.
        "mime_type": self._resolve_mime_type(metadata, filename),
    }
    if metadata.create_time is not None:
      # Left unset when the backend reports no creation time, so the model's
      # own default applies rather than a null.
      fields["create_time"] = metadata.create_time
    return ArtifactVersion(**fields)

  def _list_artifact_versions(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      filename: str,
  ) -> list[ArtifactVersion]:
    """Lists metadata for every stored version of one artifact."""
    artifact_prefix = self._get_artifact_prefix(
        app_name, user_id, filename, session_id
    )
    prefix = f"{artifact_prefix}/"
    artifact_versions = []
    for key, metadata in self._metadata_reader.read_many(
        self._fs, artifact_prefix
    ).items():
      version = _parse_version(key, prefix)
      if version is None:
        continue
      artifact_versions.append(
          self._build_artifact_version(key, version, filename, metadata)
      )
    artifact_versions.sort(key=lambda av: av.version)
    return artifact_versions

  def _get_artifact_version(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      filename: str,
      version: Optional[int] = None,
  ) -> Optional[ArtifactVersion]:
    """Gets metadata for one stored version of an artifact."""
    if version is None:
      versions = self._list_versions(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=filename,
      )
      if not versions:
        return None
      version = max(versions)

    key = self._get_object_key(
        app_name, user_id, filename, version, session_id
    )
    metadata = self._metadata_reader.read(self._fs, key)
    if metadata is None:
      return None
    return self._build_artifact_version(key, version, filename, metadata)
