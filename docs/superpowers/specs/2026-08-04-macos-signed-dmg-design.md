# Signed & Notarized macOS Build — Design Spec

**Date:** 2026-08-04
**Branch:** master
**Status:** Approved

## Problem

`pySimpleMask` currently ships Windows (`.exe` via PyInstaller) and Linux (AppImage)
release artifacts from `.github/workflows/build-releases.yml`, triggered on
`workflow_dispatch` and on pushing a `v*` tag. There is no macOS artifact.

`pysimplemask.spec`'s macOS branch (the `else` clause, since the spec only special-cases
`win32` and `linux`) builds a bare PyInstaller `EXE` — not an `.app` bundle. That's not
usable as a real macOS deliverable: it can't be code-signed as an app bundle, can't be
notarized, and isn't what users expect to drag into `/Applications`.

The user has provisioned an Apple Developer ID certificate and notarization credentials
as GitHub Actions secrets, scoped to a GitHub Environment named `macos-signing`:

```text
Secrets:  BUILD_CERTIFICATE_BASE64, P12_PASSWORD, KEYCHAIN_PASSWORD,
          APPLE_ID, APPLE_APP_PASSWORD
Variable: APPLE_TEAM_ID = S5796PM55D
```

This spec adds a signed, notarized `.dmg` (containing `pySimpleMask.app`) as a third
release artifact, built and gated the same way the existing two are.

**Non-goals / explicitly out of scope for this pass:**
- Universal2 (`arm64` + `x86_64`) builds — `macos-14` runners are Apple Silicon;
  arm64-only is sufficient for now. Revisit only if Intel Mac users become a real ask.
- A standalone `build-macos.yml` workflow (the shape suggested by the reference
  `help.txt` doc) — rejected in favor of integrating into the existing
  `build-releases.yml`, see Architecture below.
- Any change to the already-merged `mc_dev` branch content — unrelated to this spec.

## Architecture

### Where this lives: one job in `build-releases.yml`, not a new workflow

`build-releases.yml` already has the exact trigger/gating shape this needs:
`workflow_dispatch` + `push: tags: v*`, with the `github-release` publishing job
restricted to `if: github.event_name == 'push'`. Adding `macos-build` as a third job
(alongside `windows-build`, `linux-appimage-build`) and listing it in
`github-release`'s `needs:` means:

- A manual `Run workflow` dispatch builds, signs, notarizes, and smoke-tests the `.app`/
  `.dmg` **without publishing anything** — this is the "run it manually first" checkpoint
  the user wants, and it falls out of the existing `if: github.event_name == 'push'`
  guard for free. No new conditional logic needed.
- Once a manual run's `notarytool` step reports `status: Accepted`, that exact same
  workflow already includes the macOS job in tag-triggered releases — no follow-up
  wiring change required later.

A standalone workflow (as in `help.txt`) was rejected because it would duplicate the
`github-release` publish/upload logic and create a second workflow that could race with
`build-releases.yml` when updating the same GitHub Release on a tag push.

### `pysimplemask.spec` changes

The macOS `else` branch currently ends at `exe = EXE(...)`. It needs a `BUNDLE()` wrapping
that `exe` to actually produce `dist/pySimpleMask.app`:

```python
else:
    exe = EXE(..., codesign_identity=None, entitlements_file=None)  # unchanged args
    app = BUNDLE(
        exe,
        name='pySimpleMask.app',
        icon=str(repo_root / 'packaging' / 'macos' / 'icon.icns'),
        bundle_identifier='gov.anl.aps.pysimplemask',
        info_plist={
            'NSHighResolutionCapable': True,
        },
    )
```

`CFBundleShortVersionString` is intentionally left at PyInstaller's default rather than
threaded through from `setuptools_scm`/the git tag — the other two platforms don't stamp
the tag into the binary either, only into the output filename (step 14 below), so this
stays consistent with existing practice.

`codesign_identity=None` is intentional: PyInstaller does **not** sign during the build.
The workflow performs an explicit `codesign --deep --options runtime --timestamp`
afterward (see below), which gives full control over the hardened-runtime entitlements
needed for notarization — matching the `help.txt` reference flow rather than relying on
PyInstaller's built-in (shallower) signing.

`packaging/macos/icon.icns` does not exist yet and is **not** committed as a binary
asset (unlike `packaging/icon.ico`) — the workflow generates it at build time from the
existing `packaging/icon.png` via `sips`/`iconutil` (both are macOS-only tools, hence
generated in CI rather than checked in and cross-platform-buildable). A small helper
script `packaging/macos/make_icns.sh` encapsulates this so it's reviewable and testable
independent of the workflow YAML.

### Entitlements

New file `packaging/macos/entitlements.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.cs.allow-unsigned-executable-memory</key>
    <true/>
    <key>com.apple.security.cs.disable-library-validation</key>
    <true/>
</dict>
</plist>
```

Both entitlements are near-universal requirements for notarizing PyInstaller-bundled
Python/Qt apps: the bundle contains compiled extensions (numpy, scipy, PySide6's Qt
frameworks) that are not all signed by the same Team ID, and hardened runtime's default
library validation will refuse to load them at process start without
`disable-library-validation`. `allow-unsigned-executable-memory` avoids hardened-runtime
crashes from CPython's JIT-adjacent memory use. These are applied via
`codesign --entitlements packaging/macos/entitlements.plist` in the sign step, not baked
into the spec file.

### `macos-build` job — step sequence

Runs on `macos-14`, `environment: macos-signing` (required for the job to see the six
secrets/vars listed above — mirrors `help.txt` verbatim for the cert-import step since
that's Apple-tooling boilerplate, not project-specific logic):

1. Checkout (`fetch-depth: 0`, matching the other two jobs — `setuptools_scm` needs tag
   history to derive the version).
2. `actions/setup-python@v7` at `${{ env.PYTHON_VERSION }}` (3.12 — same as Windows;
   Linux pins 3.11 only because of a RockyLinux 9 package-availability constraint that
   doesn't apply here).
3. Cache PyInstaller `build/` (same cache-key pattern as the other two jobs, keyed on
   `pysimplemask.spec` + `packaging/entrypoint.py`).
4. `pip install -e ".[pyinstaller]"` (same extra already used by Windows/Linux).
5. Generate `packaging/macos/icon.icns` via `packaging/macos/make_icns.sh`.
6. Import the Developer ID certificate into a temporary keychain — verbatim from
   `help.txt`'s template (create keychain → unlock → import p12 → set
   `security set-key-partition-list` → add to search list → `find-identity` to resolve
   `SIGNING_IDENTITY` into `$GITHUB_ENV`). Fails the step with `::error::` if no
   `Developer ID Application` identity is found, so a broken cert import fails loudly
   instead of silently producing an unsigned build.
7. `pyinstaller pysimplemask.spec` → produces `dist/pySimpleMask.app`.
8. `codesign --force --deep --options runtime --timestamp --entitlements packaging/macos/entitlements.plist --sign "$SIGNING_IDENTITY" dist/pySimpleMask.app`.
9. `codesign --verify --deep --strict --verbose=2` + `codesign --display --verbose=4` —
   fail fast here rather than discovering a bad signature at notarization (which is
   slower and rate-limited).
10. Smoke-test launch: open the `.app`'s embedded binary directly (not through Launch
    Services) and confirm it's still running after ~8s, then kill it — mirrors the
    Windows job's `Start-Process` / `Stop-Process` liveness check for consistency across
    the three platform jobs.
11. Build the `.dmg`: `mkdir dmg-root`, copy the `.app` in, symlink `/Applications`,
    `hdiutil create -format UDZO`.
12. `xcrun notarytool submit ... --wait`, then `xcrun stapler staple` +
    `xcrun stapler validate`.
13. `spctl --assess --type open --context context:primary-signature` — final Gatekeeper
    check, matching `help.txt`.
14. Rename to `pySimpleMask-<tag>-macos.dmg`, matching the existing
    `pySimpleMask-<tag>-windows.exe` / `pySimpleMask-<tag>-x86_64.AppImage` naming
    convention so `github-release`'s glob-based `files:` list just needs one more line.
15. `actions/upload-artifact@v7`, `name: macos-dmg`, `retention-days: 14` — same as the
    other two jobs.
16. `if: always()` — delete the temporary keychain, regardless of prior step outcomes.

### `github-release` job changes

- Add `macos-build` to `needs:`.
- Add a `Download macOS dmg artifact` step (mirrors the two existing download steps).
- Add `dist/pySimpleMask-*-macos.dmg` to the `files:` glob list.

## Error handling

- Missing/invalid certificate → step 6 fails explicitly (`::error::` + non-zero exit)
  rather than silently falling through to an unsigned build.
- Bad signature → caught at step 9, before spending a notarization round-trip.
- `notarytool submit --wait` surfaces Apple's rejection reason directly in the log if
  notarization fails (e.g., a missing entitlement); no extra handling needed beyond
  letting the step fail the job.
- Temp keychain cleanup runs `if: always()` so a mid-pipeline failure never leaves a
  stray keychain on the (ephemeral, single-use) runner — low-stakes since the runner is
  destroyed after the job anyway, but matches `help.txt`'s hygiene.

## Testing

This spec can't be verified end-to-end outside GitHub Actions — Apple's signing and
notarization services aren't reachable or fakeable locally, and there's no macOS runner
in this environment. What's checkable ahead of a live run:
- `pysimplemask.spec` parses (PyInstaller can load it) and the macOS branch's `BUNDLE()`
  call has valid argument shapes.
- The workflow YAML is valid.
- `packaging/macos/make_icns.sh` and `entitlements.plist` are well-formed.

Real verification is the user manually dispatching `build-releases.yml` from the Actions
tab and confirming: `security find-identity` shows the `Developer ID Application` cert,
`codesign --verify` passes, `notarytool` reports `status: Accepted`, and the Gatekeeper
`spctl` assessment passes. That manual run **is** the test plan for this feature — it
can't be front-loaded into this session.

## Open risk

Entitlements requirements for notarizing PyInstaller/PySide6 apps can be
project-specific — the two entitlements included here are the common baseline, but if
the manual run's notarization or Gatekeeper step fails, the most likely next step is
tightening/loosening `entitlements.plist`, not restructuring the workflow.
