# Remote Compilation Cache

**Status**: Implemented

**Issue**: [GH-1942](https://github.com/NVIDIA/warp/issues/1942)

## Motivation

Cloud jobs with ephemeral disks recompile the same Warp kernels when each new worker starts.
A shared cache can let a job reuse artifacts compiled by an earlier job and reduce cold-start
time without requiring a persistent volume on every worker. The first use case is a Google
Cloud Storage (GCS) bucket shared by jobs running the same published Warp build.

Warp's compilers and module loaders require local files. The existing local cache also has
concurrency rules for publishing a module binary and its metadata, or a MathDx LTO artifact
and its sidecars. A remote cache must preserve those rules rather than pass a `gs://` path to
the compiler or treat a set of GCS objects as a local directory.

## Requirements

| ID | Requirement | Priority |
| --- | --- | --- |
| R1 | Reuse compiled kernel and MathDx LTO artifacts across jobs using the same published Warp build. | Must |
| R2 | Keep local cache hits and all compiler and loader paths local. | Must |
| R3 | Make a complete cache entry visible remotely as one unit, including its sidecars. | Must |
| R4 | Let concurrent producers publish without overwriting a completed entry. | Must |
| R5 | Fall back to local compilation on a remote miss, invalid archive, or storage failure. | Must |
| R6 | Distinguish incompatible module, target, toolchain, and artifact formats in remote keys. | Must |
| R7 | Keep GCS packages optional and avoid importing them when remote caching is disabled. | Must |
| R8 | Allow authenticated read-only consumers and suppress uploads for cheap compilations. | Should |
| R9 | Keep the storage boundary small enough for a later read-only backend. | Should |

**Non-goals:**

- Accepting a remote path through `WARP_CACHE_PATH` or making the compiler operate on remote
  files.
- Supporting HTTP, Artifactory, S3, Azure, or anonymous public buckets in the first release.
- Sharing entries between different Warp releases, development builds, or independently built
  copies that happen to report the same final version.
- Repairing, deleting, listing, or evicting GCS objects from Warp's normal cache path.
- Reworking the existing local cache commit algorithm as part of remote-cache integration.

## Design

### Approach

Add an optional GCS-backed **second tier** behind the existing local cache. On a local miss,
Warp looks up one remote object for the requested compilation identity. A valid object is
downloaded into a unique local staging directory, then passed through Warp's existing local
publication path. If no valid remote entry is available, Warp compiles normally. After a
successful compilation, Warp builds one archive from the committed local cache files and
attempts to create its remote object.

The remote object is the publication unit. A kernel entry contains its loadable binary and
`.meta` file; a MathDx LTO entry contains its `.lto` file and any required `.meta` or
`_fatbin.lto` sidecar. Generated source and compilation traces stay local. The archive is
finished and closed in a unique local temporary file before upload begins. GCS makes a
single object visible only after its upload completes, so no remote reader can see half of
an entry. See [Cloud Storage consistency](https://docs.cloud.google.com/storage/docs/consistency).

### Public API

Warp exposes three settings in `warp.config`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `remote_cache_dir` | `None` | A `gs://bucket/prefix` root; `None` disables remote work. |
| `remote_cache_read_only` | `False` | Permit remote reads but do not publish. |
| `remote_cache_min_compile_time` | `1.0` seconds | Publish only compilations at or above this duration. |

For a job that can read and publish cache entries, configure:

```bash
pip install "warp-lang[remote-cache-gcs]"
```

```python
import warp as wp

wp.config.remote_cache_dir = "gs://my-bucket/warp-cache"
wp.config.remote_cache_read_only = False
wp.config.remote_cache_min_compile_time = 1.0
wp.init()
```

Set `remote_cache_read_only = True` for consumers that should not publish. No new `wp.*`
function is required.

Configure the remote cache before `wp.init()`. `kernel_cache_dir` and
`WARP_CACHE_PATH` continue to select local storage only. A local hit performs no remote
operation. The first backend accepts only `gs://` roots and uses Google Application Default
Credentials available to the job. Read-only mode does not imply anonymous access. Missing
credentials cause a warning and a local-compilation fallback; Warp adds no credential field
or secret-bearing URI syntax.

Remote reads and writes are enabled only when `warp.config.version` consists of exactly three
numeric components, such as `1.19.0`. Development, release-candidate, post-release, local,
and other suffixed versions use only the local cache, even if `remote_cache_dir` is set. A
shared prefix must be used only by jobs running the same published Warp build. The version
gate deliberately does not attempt to fingerprint arbitrary source modifications to a build
that still reports a final version string.

### Entry identity and archive format

Each entry gets a SHA-256 digest of canonical JSON containing the archive-format version,
Warp version, entry kind, namespace, expected artifact names, and compilation identity.
The remote key is `gs://bucket/prefix/<warp-version>/<namespace>/<full-digest>.tar.gz`.
The digest distinguishes remote entries even though Warp keeps its existing abbreviated
local names.

Kernel identities include the full module hash and actual output names. The CPU target
identity includes LLVM version, target triple, resolved compiler flags, and host CPU name and
features when `-march=native` is used. The CUDA target identity includes output format,
SM target, architecture suffix, selected compiler and compiler version, and CUDA Toolkit
version. It omits the driver version. MathDx LTO identities include the full LTO symbol,
SM target, libmathdx and CUDA Toolkit versions, and the required artifact set. Keys are
derived from the inputs Warp uses to build each artifact, not from the worker's identity.

An entry is one gzip-compressed tar stream containing only known regular-file members and
`manifest.json`. Artifact basenames longer than 100 characters or containing non-ASCII
characters use deterministic short ASCII names inside the tar stream; the manifest and
local cache retain their original names. Archive metadata is normalized so equivalent
inputs have a stable format. The manifest records the full canonical identity, artifact
sizes, and SHA-256 hashes. On download, Warp checks the identity and every member while
streaming into known local paths; it never extracts arbitrary tar paths. The first release
caps compressed input at 1 GiB, total extracted data at 4 GiB, each artifact at 2 GiB,
and the manifest at 1 MiB. A failed check leaves no published local entry.

### Store boundary and GCS backend

The internal storage interface needs only an exact-key binary reader and a create-if-absent
upload from a completed local file. It does not expose filesystem operations such as rename,
directory listing, or recursive copy. The initial implementation uses the optional
`google-cloud-storage` Python package, imported lazily through a `remote-cache-gcs` extra.
The ordinary Warp installation gains no GCS dependency. The client is initialized only for
an eligible, configured transfer and recreated after `fork()`.

The GCS adapter constructs an object from its `gs://` URI, reloads its metadata, and passes
the observed generation as a precondition on each chunked read. It translates a missing
object into a cache miss. Pinning prevents a read from mixing chunks from different versions
if an operator replaces an object.
The adapter uploads the closed archive with `if_generation_match=0`. If two jobs publish the
same key, one wins and the other keeps its local result. The SDK exposes this precondition
directly. See [Blob operations](https://docs.cloud.google.com/python/docs/reference/storage/latest/google.cloud.storage.blob.Blob)
and [GCS request preconditions](https://docs.cloud.google.com/storage/docs/request-preconditions).

Remote transfers have finite internal request and retry limits. A missing entry is an
ordinary, quiet miss. Authentication, network, timeout, and validation errors produce
bounded warnings and fall back to local compilation. Failure to upload never invalidates
the completed local cache entry. An upload rejected because the object already exists is a
normal race loss, not a warning. Warp does not automatically delete a bad existing object;
an operator can remove it, after which a later compilation may recreate it.

### Local publication and concurrency

The local cache remains authoritative. For kernels, Warp first tries to rename a completed
process-specific build directory into the cache. When the destination already exists, its
current fallback moves the binary and metadata separately. Remote restoration uses a unique
staging directory on the same filesystem, validates the whole entry, then follows the same
binary-first, metadata-last publication rules. MathDx LTO restoration goes through the
existing `_build_lto_base()` commit block and its sidecar checks.

Upload archives are made from the **committed local cache files**, after Warp has established
the required complete entry. This treats Warp's local cache as the source of truth. A defect
in local publication should be fixed in that shared path so local-only and remote-enabled
jobs both benefit. Remote caching is bypassed when compilation targets a custom output
directory. Modules with `strip_hash=True` are excluded because their stable local names can
be replaced by a different module identity.

GCS atomicity applies to one object, not to the local binary and metadata moves. This design
does not upload those files as separate GCS objects or depend on a bucket-specific folder
rename feature. Concurrent downloads may stage the same entry independently; they do not
hold a transfer-wide lock. The local commit path resolves their race.

### Trust and retention

Compiled cache entries are executable artifacts. Jobs must trust all principals with write
access to their GCS prefix. Archive checks detect malformed or damaged entries; they do not
make a malicious writer safe. Prefixes should be private to the intended published build and
managed with appropriate GCS permissions.

`wp.clear_kernel_cache()` and `wp.clear_lto_cache()` remain local-only. Bucket lifecycle
rules or an operator manage remote retention and removal of bad entries. No automatic
remote repair is attempted after a validation failure. The SDK may attempt to clean up an
upload that fails its own checksum check; this is distinct from repairing an existing entry.

### Alternatives considered

**Use `gcsfs` or Etils as a path abstraction.** These libraries offer convenient
filesystem-like APIs, but neither makes several GCS objects an atomic cache entry. Warp
needs exact-key reads and conditional creation of one object; the Google SDK exposes the
generation preconditions and read version directly. The choice is based on those operations,
not on dependency count. Etils's GCS path extra also delegates to `gcsfs`.

**Point Warp's existing cache directory at a GCSFuse mount.** This can work for deployments
that provide a mount. It makes cache correctness depend on mount behavior and availability,
while the motivating cloud jobs may have only local scratch storage. A native second tier
preserves Warp's local compiler and loader behavior.

**Upload each output file separately.** A binary could become visible before its metadata or
LTO sidecar, requiring an additional commit marker and cleanup protocol. One archive gives
the entry a single publication point.

**Upload asynchronously.** This would reduce the producing job's latency, but introduces
shutdown, retry, and process-lifetime questions. The first release publishes synchronously
after local compilation, and the compile-time threshold limits inexpensive uploads.

## Testing Strategy

Use `unittest` and an in-memory store to cover final-version gating, canonical keys,
archive validation and size limits, local-first behavior, read-only mode, publication
thresholds, failures, and concurrent producers and consumers. Test the kernel and MathDx LTO
paths with separate local cache roots and verify that a remote consumer can use a complete
entry without invoking compilation. Test local publication races with barriers and exact
artifact assertions rather than timing expectations.

Run GCS adapter tests against a pinned `fake-gcs-server` container in both GitHub and GitLab
CI. Cover missing objects, generation preconditions passed to each read chunk, interrupted
or invalid downloads, and create-only uploads on both sides of the SDK's resumable-upload
threshold. Two producers using the same remote key must leave one object that a third
consumer can restore. The emulator uses `STORAGE_EMULATOR_HOST` and requires no production
credentials. It does not enforce read preconditions after object replacement, so the read
test checks the SDK arguments rather than the server's response.

Development checkouts cannot satisfy the final-version gate. Tests may patch the private
eligibility predicate while leaving Warp's version string unchanged; production code has no
override. CUDA and MathDx tests run where available, while CPU and archive tests remain
useful on systems without a GPU.
