# Remote Compilation Cache

**Status**: In Progress

**Issue**: [GH-1942](https://github.com/NVIDIA/warp/issues/1942)

## Motivation

Cloud jobs with ephemeral disks recompile the same Warp kernels when each new worker starts.
A shared cache lets a job reuse kernels compiled by an earlier job and shortens cold starts
without requiring a persistent volume on every worker. The request came from jobs that share
a Google Cloud Storage (GCS) bucket, but the capability is not specific to Google Cloud.

Warp's compilers and module loaders require local files, and the local cache has its own
rules for publishing a module binary and its metadata under concurrent compiles. A remote
cache must keep those paths local rather than pass a `gs://` path to the compiler.

## Requirements

| ID | Requirement | Priority |
| --- | --- | --- |
| R1 | Reuse compiled kernels across jobs that run the same Warp build. | Must |
| R2 | Keep local cache hits and all compiler and loader paths local. | Must |
| R3 | Use one cache layout: a remote root is a copy of a local cache directory. | Must |
| R4 | Never let a reader load an incomplete entry, including under concurrent publishers. | Must |
| R5 | Fall back to local compilation on a remote miss or storage failure, at bounded cost. | Must |
| R6 | Keep storage backends behind a small interface, and import a backend's packages only when its root is configured. | Must |
| R7 | Allow read-only consumers and suppress uploads of cheap compilations. | Should |

**Non-goals:**

- Accepting a remote path through `WARP_CACHE_PATH` or making the compiler operate on remote
  files.
- Caching MathDx LTO objects or generated source remotely.
- Defining a second cache key or artifact format for remote entries.
- Sharing entries between different Warp versions, or from source checkouts.
- Listing, repairing, deleting, or evicting remote objects.
- Backends other than a directory and GCS in the first release.

## Design

### Approach

Add an optional **second tier** behind the existing local cache. On a local miss, Warp looks
for the module's binary and metadata file in the remote cache. If both are there, it downloads
them into a private staging directory and commits them through the same binary-first ordering
a local compile uses. Otherwise it compiles normally, then uploads the committed local files.

The remote cache **mirrors the local cache**. A remote root has the layout of the directory
named by `WARP_CACHE_PATH`, and each cached file is one remote object under its local name:

```
<root>/<Warp version>/wp_<module>_<hash>/wp_<module>_<hash>.sm86.cubin
<root>/<Warp version>/wp_<module>_<hash>/wp_<module>_<hash>.meta
```

A copy of a local cache directory is therefore a valid remote root, and a copy of a remote root
is a valid local cache. Operators can fill, inspect, and prune a remote cache with ordinary
tools such as `gcloud storage rsync`, and the same tree can back a mounted directory or, later,
a read-only HTTP mirror.

### Public API

Warp exposes three settings in `warp.config`, read once by `wp.init()`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `remote_cache_dir` | `None` | Remote root: a directory or a `gs://bucket/prefix` URL. `None` disables remote work. |
| `remote_cache_read_only` | `False` | Fetch from the remote cache but do not publish. |
| `remote_cache_min_compile_time` | `1.0` seconds | Publish only modules that took at least this long to compile. |

```python
import warp as wp

wp.config.remote_cache_dir = "gs://my-bucket/warp-cache"
wp.init()
```

`kernel_cache_dir` and `WARP_CACHE_PATH` continue to select local storage only. A local hit
performs no remote operation. No new `wp.*` function is required.

The remote cache is enabled only when `warp.config.version` names one published build: a
release such as `1.19.0` or a date-stamped nightly build such as `1.20.0.dev20261118`. A source
checkout reports `.dev0` whatever it contains, so it uses only the local cache and warns that
the setting was ignored.

### Entry identity

A remote entry has exactly the identity of its local file names. Those names encode:

- the Warp version, as the top-level directory;
- the module name and the first seven hex digits of the module hash, which covers the kernel
  source and the resolved module options;
- for CUDA binaries, the target architecture, architecture suffix, and output format, as in
  `.sm90a.cubin` or `.sm86.ptx`;
- for CPU objects, a hash of the LLVM version, the target triple, and, with `-march=native`,
  the host's ISA features, as in `.cpu1a2b3c4d.o`.

Two things are not encoded. The module hash is abbreviated, and the CUDA Toolkit version is
absent, so a CUDA 12 and a CUDA 13 build of one Warp version share entries even though a kernel
compiled with the newer toolkit can fail to load on an older driver. Both gaps exist in the
local cache today whenever `WARP_CACHE_PATH` names a directory shared between machines. The
remote cache deliberately inherits them: a remote root must be shared only among jobs running
the same Warp build, and any gap worth closing should be closed in the local names so that
both tiers benefit. A separate remote key would give Warp two definitions of cache identity
to keep in agreement.

The remote tier is skipped for builds whose local name does not identify their contents or
that do not land in the kernel cache: modules with `strip_hash=True`, compiles into a custom
output directory, forced rebuilds for autograd array-access verification, and any compile
while `warp.config.cache_kernels` is disabled.

### Completeness without a commit marker

An entry is two objects, and they do not become visible together. This is safe for the same
reason it is safe in the local cache, where the files are also committed one at a time:

- A reader requires both files and treats the absence of either as a miss.
- Each object becomes visible only once complete. This is part of the store interface.
- The metadata file belongs to the module directory, not to one binary. It is shared by every
  target compiled into that directory, exactly as in the local cache.

Every intermediate state is therefore a miss, and a miss costs only a local compilation, whose
publication then completes the entry. No state lets a reader load a partial entry.

### Store interface and backends

The core module `warp/_src/remote_cache.py` knows nothing about a particular service. A backend
implements two methods over keys relative to the remote root:

```python
class RemoteStore(Protocol):
    def get(self, key: str, path: Path) -> bool: ...  # False if the object is absent
    def put(self, path: Path, key: str) -> None: ...  # no-op if the object exists
```

A backend must make each object visible only once it is complete and must not replace an
existing object. It needs no listing, rename, delete, or conditional-read operation, which
keeps object stores, artifact repositories, and plain file servers within reach. The backend
is chosen from the root: a string without a URL scheme is a directory, and a URL is looked up
by scheme. An unsupported scheme disables the remote cache with a warning.

**Directory.** Files are copied to and from a directory tree, such as a network mount or a
bucket mounted through a FUSE driver. Uploads are copied under a private name and renamed into
place. This backend has no dependencies and is the one the core tests run against.

**Google Cloud Storage.** `warp/_src/remote_cache_gcs.py` uses the optional
`google-cloud-storage` package, installed through the `remote-cache-gcs` extra and imported
only when a `gs://` root is first used. It authenticates with Application Default Credentials
and adds no credential setting. Uploads pass `if_generation_match=0`, so they create objects
without replacing them and need no delete permission on the bucket. See
[GCS request preconditions](https://docs.cloud.google.com/storage/docs/request-preconditions).

The store is created on first use, not in `wp.init()`, and is recreated after `fork()`.

### Failure handling

A missing object is an ordinary, quiet miss. Any other read failure, such as missing
credentials, a network timeout, or a missing optional package, produces one warning and
disables the remote cache for the rest of the process. A write failure produces one warning
and disables publishing, but fetching continues, which is the useful behavior for a job that
has read-only credentials and did not set `remote_cache_read_only`. A broken remote therefore
costs a job at most one failed request in each direction, however many modules it compiles.
Neither failure affects the local cache entry.

### Local publication and concurrency

The local cache remains authoritative. A restored entry is downloaded into a unique staging
directory inside the kernel cache directory, so that the commit is a rename on one filesystem,
and then committed binary first and metadata last, without replacing files a concurrent
process has already committed. Concurrent downloads of one entry stage independently, and the
local commit resolves their race.

Uploads read the **committed local files** after a compile, which may be the equivalent output
of a concurrent compile that committed first. Publication is synchronous. The compile-time
threshold skips modules that are cheap to recompile: in a 1,242-module MuJoCo Warp test-suite
cache, 85% of modules compiled in under one second but accounted for only 18% of compile time.

### MathDx LTO objects

LTO objects for tile operations are inputs to kernel compilation, and a compiled kernel
already contains the MathDx code it links. A remote kernel hit returns before code generation
and never needs them. They would help only when a kernel misses while an LTO object it needs
is available, and rebuilding one is comparable to the compile it feeds: about half a second
for a 16x16 Cholesky factorization in one measurement. They are left out of the first release.
Because the layout is mirrored, adding the `lto/` directory later needs no new format.

### Trust and retention

Compiled cache entries are executable code. Jobs must trust every principal with write access
to their remote root. Warp verifies nothing about a restored file beyond what the storage
layer checks in transit.

`wp.clear_kernel_cache()` and `wp.clear_lto_cache()` remain local-only. Warp never replaces
or deletes a remote object. Bucket lifecycle rules or an operator manage retention and remove
bad entries, after which a later compilation recreates them.

### Alternatives considered

**One archive object per entry, keyed by a digest of the compile identity.** A prototype
stored each entry as a `tar.gz` with a manifest under a SHA-256 key covering the full module
hash, compiler, and toolkit version. One object gave a single publication point and a
collision-proof key. It also introduced a second definition of cache identity, a versioned
archive format with its own validation and size limits, and a bucket that could not be
filled, inspected, or mounted with ordinary tools. Its atomicity was not needed, as described
above, and its archive hardening guarded against writers that the trust model already
requires to be trusted.

**Synchronize the cache directory around each job.** Copying the cache from the bucket before
a run and back afterwards needs no change to Warp. It transfers the whole cache in both
directions, needs hooks at job start and exit, and loses a preempted job's work. The mirrored
layout keeps this workflow compatible with the remote tier: either can fill a root the other
reads.

**Point `WARP_CACHE_PATH` at a GCSFuse mount.** This makes compilation and module loading
depend on the mount's behavior and performance. Where a mount exists, it can instead be the
remote root, through the directory backend, while compilation stays on local disk.

**Use `gcsfs` or Etils as the only storage layer.** These provide many backends through one
path abstraction, at the cost of a larger optional dependency. The store interface is small
enough that such a backend can be added beside the existing ones if users need it.

**Upload asynchronously.** This would reduce the producing job's latency but raises shutdown,
retry, and process-lifetime questions. The compile-time threshold limits the cost instead.

### Open questions

These were put to the requesters in the
[issue](https://github.com/NVIDIA/warp/issues/1942#issuecomment-6989746258) and may change
which backends ship:

- How their jobs reach a bucket: the `google-cloud-storage` package, `etils.epath`, `gcsfs`,
  or an internal library. The answer decides whether the GCS backend stays as it is, is
  replaced, or is joined by a way to register a store from user code.
- Whether every job should publish or one trusted job should fill the cache.
- Whether leaving MathDx LTO objects out is acceptable.

## Testing Strategy

Core tests use `unittest` and the real directory backend in temporary directories. They cover
the mirrored layout, partial entries, the publication threshold, read-only mode, version
eligibility, unsupported roots, and the failure breakers, using small failing stores only for
the last. Kernel-level tests compile with one local cache, restore into a second without
running code generation, confirm that a copy of the remote root works as a local cache, and
confirm that nonstandard builds neither fetch nor publish. A cross-process test runs a producer
and a read-only consumer through `wp.launch()`, so the restored binary is loaded and executed.
The local commit race is tested with a barrier and exact file assertions rather than timing.

GCS backend tests run the backend against an in-memory stand-in for the storage client that
raises the SDK's own exception types, so they need the `remote-cache-gcs` extra but no
network, credentials, or storage service. They cover a missing object, object naming under
the root prefix, uploads that must not replace an existing object, service errors reaching
the caller, and a `gs://` root reaching the backend through the core module. One Linux x86_64
test job in each of GitHub and GitLab CI installs the extra; the tests are skipped wherever
it is absent. A storage emulator is not used because it would add a dependency that has not
been through open source dependency review. The backend has not been run against real GCS, so
the SDK's transfer behavior itself is untested here.

A source checkout cannot pass the version check, so tests patch the private eligibility
predicate while leaving Warp's version string unchanged. Production code has no override.
