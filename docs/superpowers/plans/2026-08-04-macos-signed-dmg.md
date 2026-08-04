# Signed & Notarized macOS Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a signed, notarized macOS `.dmg` (containing `pySimpleMask.app`) as a
third release artifact alongside the existing Windows `.exe` and Linux AppImage.

**Architecture:** Extend the existing `.github/workflows/build-releases.yml` with a new
`macos-build` job (rather than a standalone workflow), gated by the `macos-signing`
GitHub Environment. `pysimplemask.spec`'s macOS branch gains a `BUNDLE()` call to
actually produce an `.app` (it currently only builds a bare `EXE`). The job imports the
Developer ID certificate, builds, deep-signs with hardened runtime, notarizes via
`notarytool`, staples, and uploads the `.dmg`; `github-release` picks it up like the
other two platforms.

**Tech Stack:** GitHub Actions (`macos-14` runner), PyInstaller, Apple `codesign` /
`notarytool` / `hdiutil` / `spctl`, pytest, PyYAML (test-only).

**Reference spec:** `docs/superpowers/specs/2026-08-04-macos-signed-dmg-design.md`

## Global Constraints

- GitHub Environment name is exactly `macos-signing` (already provisioned by the user
  with secrets `BUILD_CERTIFICATE_BASE64`, `P12_PASSWORD`, `KEYCHAIN_PASSWORD`,
  `APPLE_ID`, `APPLE_APP_PASSWORD` and variable `APPLE_TEAM_ID`).
- Bundle identifier: `gov.anl.aps.pysimplemask`.
- Target architecture: arm64 only (`macos-14` runner), no universal2 build.
- Integrate into `build-releases.yml` — do not create a standalone workflow file.
- Artifact naming: `pySimpleMask-<tag>-macos.dmg`, matching the existing
  `pySimpleMask-<tag>-windows.exe` / `pySimpleMask-<tag>-x86_64.AppImage` convention.
- A manual `workflow_dispatch` run must build/sign/notarize/smoke-test but must NOT
  publish a GitHub Release — this already falls out of `github-release`'s existing
  `if: github.event_name == 'push'` guard; do not add new conditionals to defeat this.
- Action versions must match what's already used elsewhere in the file
  (`actions/checkout@v7`, `actions/setup-python@v7`, `actions/cache@v5`,
  `actions/upload-artifact@v7`, `actions/download-artifact@v8`).
- Never print `BUILD_CERTIFICATE_BASE64`, `P12_PASSWORD`, `KEYCHAIN_PASSWORD`,
  `APPLE_ID`, or `APPLE_APP_PASSWORD` in logs. (`APPLE_TEAM_ID` is a `vars.` value, not
  a secret — it's already public in every signed binary via `codesign -dv`, and
  `security find-identity`/`codesign --display` legitimately print it; the final
  whole-branch review confirmed this isn't a leak.)
- No macOS runner and no real Apple credentials exist in this dev environment — tasks
  below test everything that's testable without them (YAML structure, spec branch
  logic, plist/shell syntax) and call out explicitly what can only be verified via a
  live `workflow_dispatch` run.

---

### Task 1: Confirm `master` already contains `mc_dev`

**Files:** none (verification only).

**Interfaces:** none.

- [ ] **Step 1: Re-check branch state**

```bash
git fetch origin
git rev-list --left-right --count master...origin/master
git log --oneline master..mc_dev
```

Expected: first command prints `0	0` (local `master` matches `origin/master`); second
command prints nothing (no commits exist in `mc_dev` that aren't already in `master` —
`mc_dev` was merged via PR #29, commit `ca9c3c2`).

- [ ] **Step 2: If step 1's second command printed anything, merge it**

Only do this if `git log --oneline master..mc_dev` was non-empty:

```bash
git checkout master
git merge mc_dev
```

Otherwise skip — `master` is already up to date and there is nothing to merge or
commit for this task.

---

### Task 2: Add macOS icon generation script and entitlements

**Files:**
- Create: `packaging/macos/make_icns.sh`
- Create: `packaging/macos/entitlements.plist`
- Create: `tests/packaging/__init__.py`
- Create: `tests/packaging/test_macos_packaging_files.py`

**Interfaces:**
- Produces: `packaging/macos/make_icns.sh` (invoked by the workflow in Task 4; reads
  `packaging/icon.png`, writes `packaging/macos/icon.icns`).
- Produces: `packaging/macos/entitlements.plist` (referenced by the `codesign` step
  added in Task 4).

- [ ] **Step 1: Write the failing tests**

```python
# tests/packaging/test_macos_packaging_files.py
"""Static checks for the macOS packaging assets.

Neither `sips`/`iconutil` (used by make_icns.sh) nor real icon generation can
run on this dev machine's Linux/pytest environment, so these tests only check
the two things that ARE verifiable without a macOS runner: the shell script
is syntactically valid, and the entitlements file is a well-formed plist with
the two keys notarization needs.
"""

import plistlib
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKE_ICNS = REPO_ROOT / "packaging" / "macos" / "make_icns.sh"
ENTITLEMENTS = REPO_ROOT / "packaging" / "macos" / "entitlements.plist"


def test_make_icns_script_exists_and_is_executable_bash():
    assert MAKE_ICNS.exists()
    result = subprocess.run(
        ["bash", "-n", str(MAKE_ICNS)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_entitlements_plist_is_well_formed():
    with open(ENTITLEMENTS, "rb") as f:
        data = plistlib.load(f)
    assert data["com.apple.security.cs.allow-unsigned-executable-memory"] is True
    assert data["com.apple.security.cs.disable-library-validation"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/packaging/test_macos_packaging_files.py -v`
Expected: both tests FAIL — `MAKE_ICNS`/`ENTITLEMENTS` don't exist yet
(`FileNotFoundError` from `subprocess.run` / `open`).

- [ ] **Step 3: Create `packaging/macos/entitlements.plist`**

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

- [ ] **Step 4: Create `packaging/macos/make_icns.sh`**

```bash
#!/usr/bin/env bash
# Generates packaging/macos/icon.icns from packaging/icon.png.
#
# Uses sips/iconutil, which only exist on macOS, so this runs in CI
# (macos-14 runner) rather than being committed as a prebuilt binary asset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SOURCE_PNG="$REPO_ROOT/packaging/icon.png"
ICONSET_DIR="$REPO_ROOT/packaging/macos/icon.iconset"
OUTPUT_ICNS="$REPO_ROOT/packaging/macos/icon.icns"

rm -rf "$ICONSET_DIR"
mkdir -p "$ICONSET_DIR"

for size in 16 32 128 256 512; do
  sips -z "$size" "$size" "$SOURCE_PNG" \
    --out "$ICONSET_DIR/icon_${size}x${size}.png" >/dev/null
  double=$((size * 2))
  sips -z "$double" "$double" "$SOURCE_PNG" \
    --out "$ICONSET_DIR/icon_${size}x${size}@2x.png" >/dev/null
done

iconutil -c icns "$ICONSET_DIR" -o "$OUTPUT_ICNS"
rm -rf "$ICONSET_DIR"

echo "Wrote $OUTPUT_ICNS"
```

Make it executable: `chmod +x packaging/macos/make_icns.sh`

- [ ] **Step 5: Create `tests/packaging/__init__.py`**

Empty file, matching the pattern of `tests/core/__init__.py`, `tests/cli/__init__.py`,
etc.

- [ ] **Step 6: Run tests to verify they pass**

Run: `/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/packaging/test_macos_packaging_files.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add packaging/macos/make_icns.sh packaging/macos/entitlements.plist \
        tests/packaging/__init__.py tests/packaging/test_macos_packaging_files.py
git commit -m "feat(packaging): add macOS icon generation script and entitlements"
```

---

### Task 3: Wrap the macOS PyInstaller `EXE` in a `BUNDLE()`

**Files:**
- Modify: `pysimplemask.spec:144-163`
- Create: `tests/packaging/test_pysimplemask_spec.py`

**Interfaces:**
- Consumes: `packaging/macos/icon.icns` (path only — referenced as a string, not read,
  so this task doesn't need Task 2's script to have actually run).
- Produces: when built with PyInstaller on macOS, `dist/pySimpleMask.app` with
  `CFBundleIdentifier=gov.anl.aps.pysimplemask` — consumed by Task 4's codesign/dmg
  steps, which operate on `dist/pySimpleMask.app` (not the bare `dist/pySimpleMask`
  binary the old code produced).

- [ ] **Step 1: Write the failing test**

```python
# tests/packaging/test_pysimplemask_spec.py
"""Exercises pysimplemask.spec's per-platform branches without needing a real
PyInstaller install or a macOS/Windows machine.

PyInstaller runs .spec files with exec(), injecting SPECPATH and the
Analysis/PYZ/EXE/BUNDLE/COLLECT builder names into the execution namespace.
We reproduce that here with recording stand-ins so the branch logic (which
builder gets which name/icon/bundle id) has real test coverage instead of
only being checkable by actually building on each target OS.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_SOURCE = (REPO_ROOT / "pysimplemask.spec").read_text()


class _Recorder:
    """Stand-in for a PyInstaller builder class; records every call."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return object()


class _AnalysisStub:
    """Stand-in for PyInstaller's Analysis result.

    The spec accesses a.pure (for PYZ) and a.scripts/a.binaries/a.datas (for
    EXE/COLLECT) across its three platform branches, so all four must exist.
    """

    def __init__(self, *args, **kwargs):
        self.pure = []
        self.scripts = []
        self.binaries = []
        self.datas = []


def _run_spec(monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)
    exe_recorder = _Recorder()
    bundle_recorder = _Recorder()
    collect_recorder = _Recorder()
    namespace = {
        "SPECPATH": str(REPO_ROOT),
        "Analysis": _AnalysisStub,
        "PYZ": lambda *a, **k: object(),
        "EXE": exe_recorder,
        "BUNDLE": bundle_recorder,
        "COLLECT": collect_recorder,
    }
    exec(compile(SPEC_SOURCE, "pysimplemask.spec", "exec"), namespace)
    return exe_recorder, bundle_recorder, collect_recorder


def test_macos_branch_wraps_exe_in_app_bundle(monkeypatch):
    exe_rec, bundle_rec, collect_rec = _run_spec(monkeypatch, "darwin")

    assert len(exe_rec.calls) == 1
    assert len(bundle_rec.calls) == 1
    assert len(collect_rec.calls) == 0

    _, exe_kwargs = exe_rec.calls[0]
    assert exe_kwargs["codesign_identity"] is None

    _, bundle_kwargs = bundle_rec.calls[0]
    assert bundle_kwargs["name"] == "pySimpleMask.app"
    assert bundle_kwargs["bundle_identifier"] == "gov.anl.aps.pysimplemask"
    assert bundle_kwargs["icon"].endswith("packaging/macos/icon.icns")


def test_windows_branch_unchanged_no_bundle(monkeypatch):
    exe_rec, bundle_rec, collect_rec = _run_spec(monkeypatch, "win32")

    assert len(exe_rec.calls) == 1
    assert len(bundle_rec.calls) == 0
    _, exe_kwargs = exe_rec.calls[0]
    assert exe_kwargs["icon"].endswith("packaging/icon.ico")


def test_linux_branch_unchanged_uses_collect(monkeypatch):
    exe_rec, bundle_rec, collect_rec = _run_spec(monkeypatch, "linux")

    assert len(exe_rec.calls) == 1
    assert len(collect_rec.calls) == 1
    assert len(bundle_rec.calls) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/packaging/test_pysimplemask_spec.py -v`
Expected: `test_macos_branch_wraps_exe_in_app_bundle` FAILS with
`assert len(bundle_rec.calls) == 1` → `0 == 1` (no `BUNDLE()` call exists yet). The
other two tests PASS already since they test the unchanged win32/linux branches.

- [ ] **Step 3: Modify `pysimplemask.spec`'s macOS branch**

Replace this block (currently lines 144-163, the final `else:` clause):

```python
else:
    # macOS or other — single-dir EXE
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name='pySimpleMask',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
```

with:

```python
else:
    # macOS or other — single-dir EXE wrapped in an .app bundle.
    # codesign_identity stays None: the release workflow does an explicit
    # `codesign --deep --options runtime` pass afterward so it can apply the
    # hardened-runtime entitlements notarization needs.
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name='pySimpleMask',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    app = BUNDLE(
        exe,
        name='pySimpleMask.app',
        icon=str(spec_dir / 'packaging' / 'macos' / 'icon.icns'),
        bundle_identifier='gov.anl.aps.pysimplemask',
        info_plist={
            'NSHighResolutionCapable': True,
        },
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/packaging/test_pysimplemask_spec.py -v`
Expected: all three tests PASS.

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests -q`
Expected: PASS, same count as before plus the new packaging tests (no existing test
imports or executes `pysimplemask.spec`, so no regressions expected).

- [ ] **Step 6: Commit**

```bash
git add pysimplemask.spec tests/packaging/test_pysimplemask_spec.py
git commit -m "feat(packaging): bundle macOS PyInstaller build into a .app"
```

---

### Task 4: Add the `macos-build` job and wire it into `github-release`

**Files:**
- Modify: `pyproject.toml` (add `pyyaml` to the `dev` optional-dependencies list)
- Modify: `.github/workflows/build-releases.yml`
- Create: `tests/packaging/test_build_releases_workflow.py`

**Interfaces:**
- Consumes: `packaging/macos/make_icns.sh`, `packaging/macos/entitlements.plist`
  (Task 2); `dist/pySimpleMask.app` produced by `pyinstaller pysimplemask.spec`
  (Task 3).
- Produces: artifact `macos-dmg` containing `dist/pySimpleMask-<tag>-macos.dmg`,
  downloaded and published by the `github-release` job.

- [ ] **Step 1: Write the failing tests**

```python
# tests/packaging/test_build_releases_workflow.py
"""Structural checks for .github/workflows/build-releases.yml.

These don't run the workflow itself (that needs GitHub Actions plus real
Apple signing/notarization credentials, neither of which exist here) — they
catch YAML mistakes and job-wiring regressions, e.g. forgetting to add
macos-build to github-release's `needs:`, without waiting on a live run.
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "build-releases.yml"


def _load_workflow():
    with open(WORKFLOW_PATH) as f:
        return yaml.safe_load(f)


def test_workflow_is_valid_yaml_with_jobs():
    workflow = _load_workflow()
    assert "jobs" in workflow


def test_macos_build_job_targets_macos14_and_signing_environment():
    workflow = _load_workflow()
    macos_job = workflow["jobs"]["macos-build"]
    assert macos_job["runs-on"] == "macos-14"
    assert macos_job["environment"] == "macos-signing"


def test_github_release_depends_on_macos_build():
    workflow = _load_workflow()
    release_job = workflow["jobs"]["github-release"]
    assert "macos-build" in release_job["needs"]


def test_github_release_downloads_macos_dmg_artifact():
    workflow = _load_workflow()
    release_job = workflow["jobs"]["github-release"]
    download_steps = [
        step
        for step in release_job["steps"]
        if step.get("uses", "").startswith("actions/download-artifact")
    ]
    names = {step["with"]["name"] for step in download_steps}
    assert "macos-dmg" in names


def test_github_release_uploads_macos_dmg_pattern():
    workflow = _load_workflow()
    release_job = workflow["jobs"]["github-release"]
    release_step = next(
        step
        for step in release_job["steps"]
        if step.get("name") == "Create/update GitHub Release"
    )
    assert "dist/pySimpleMask-*-macos.dmg" in release_step["with"]["files"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/packaging/test_build_releases_workflow.py -v`
Expected: all FAIL — `KeyError: 'macos-build'` (job doesn't exist yet), and the
`pyyaml` import itself will fail unless already present in the env (see Step 3).

- [ ] **Step 3: Add `pyyaml` to `pyproject.toml`'s dev extras**

In `pyproject.toml`, change:

```toml
dev = [
  "coverage", # Testing
  "mypy",     # Type checking
  "pytest",   # Testing
  "ruff",     # Linting
]
```

to:

```toml
dev = [
  "coverage", # Testing
  "mypy",     # Type checking
  "pytest",   # Testing
  "pyyaml",   # Testing (workflow YAML structural checks)
  "ruff",     # Linting
]
```

(It's already importable in the project conda env, but wasn't a declared dependency;
this makes the new tests reproducible in a clean install.)

- [ ] **Step 4: Add the `macos-build` job to `.github/workflows/build-releases.yml`**

Insert a new job between `linux-appimage-build` and `github-release`:

```yaml
  macos-build:
    name: Build, sign, and notarize macOS app
    runs-on: macos-14
    environment: macos-signing
    steps:
      - uses: actions/checkout@v7
        with:
          fetch-depth: 0

      - uses: actions/setup-python@v7
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Cache PyInstaller build
        uses: actions/cache@v5
        with:
          path: build/
          key: pyinstaller-macos-${{ hashFiles('pysimplemask.spec', 'packaging/entrypoint.py') }}
          restore-keys: |
            pyinstaller-macos-

      - name: Install project and PyInstaller
        run: |
          python -m pip install --upgrade pip
          python -m pip install -e ".[pyinstaller]"

      - name: Generate .icns app icon
        run: bash packaging/macos/make_icns.sh

      - name: Import Developer ID certificate
        env:
          BUILD_CERTIFICATE_BASE64: ${{ secrets.BUILD_CERTIFICATE_BASE64 }}
          P12_PASSWORD: ${{ secrets.P12_PASSWORD }}
          KEYCHAIN_PASSWORD: ${{ secrets.KEYCHAIN_PASSWORD }}
        run: |
          CERTIFICATE_PATH="$RUNNER_TEMP/developer-id.p12"
          KEYCHAIN_PATH="$RUNNER_TEMP/signing.keychain-db"

          printf '%s' "$BUILD_CERTIFICATE_BASE64" \
            | base64 --decode > "$CERTIFICATE_PATH"

          security create-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN_PATH"
          security set-keychain-settings -lut 21600 "$KEYCHAIN_PATH"
          security unlock-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN_PATH"

          security import "$CERTIFICATE_PATH" \
            -k "$KEYCHAIN_PATH" \
            -P "$P12_PASSWORD" \
            -A \
            -t cert \
            -f pkcs12

          security set-key-partition-list \
            -S apple-tool:,apple:,codesign: \
            -s \
            -k "$KEYCHAIN_PASSWORD" \
            "$KEYCHAIN_PATH"

          security list-keychains -d user -s "$KEYCHAIN_PATH"
          security default-keychain -d user -s "$KEYCHAIN_PATH"

          echo "Available signing identities:"
          security find-identity -v -p codesigning "$KEYCHAIN_PATH"

          SIGNING_IDENTITY="$(
            security find-identity -v -p codesigning "$KEYCHAIN_PATH" \
              | grep "Developer ID Application" \
              | head -1 \
              | awk '{print $2}'
          )"

          if [ -z "$SIGNING_IDENTITY" ]; then
            echo "::error::Developer ID Application identity not found"
            exit 1
          fi

          echo "SIGNING_IDENTITY=$SIGNING_IDENTITY" >> "$GITHUB_ENV"
          echo "KEYCHAIN_PATH=$KEYCHAIN_PATH" >> "$GITHUB_ENV"

      - name: Build application
        run: pyinstaller pysimplemask.spec

      - name: Sign application bundle
        run: |
          codesign \
            --force \
            --deep \
            --options runtime \
            --timestamp \
            --entitlements packaging/macos/entitlements.plist \
            --sign "$SIGNING_IDENTITY" \
            "dist/pySimpleMask.app"

      - name: Verify application signature
        run: |
          codesign --verify --deep --strict --verbose=2 "dist/pySimpleMask.app"
          codesign --display --verbose=4 "dist/pySimpleMask.app"

      - name: Smoke test app launches
        run: |
          "dist/pySimpleMask.app/Contents/MacOS/pySimpleMask" &
          APP_PID=$!
          sleep 8
          if kill -0 $APP_PID 2>/dev/null; then
            echo "Smoke test passed: app is running after 8s"
            kill $APP_PID
          else
            wait $APP_PID || true
            echo "Smoke test FAILED: app exited early"
            exit 1
          fi

      - name: Create DMG
        run: |
          mkdir -p dmg-root
          cp -R "dist/pySimpleMask.app" dmg-root/
          ln -s /Applications dmg-root/Applications

          hdiutil create \
            -volname "pySimpleMask" \
            -srcfolder dmg-root \
            -format UDZO \
            -ov \
            "dist/pySimpleMask.dmg"

      - name: Notarize and staple DMG
        env:
          APPLE_ID: ${{ secrets.APPLE_ID }}
          APPLE_APP_PASSWORD: ${{ secrets.APPLE_APP_PASSWORD }}
          APPLE_TEAM_ID: ${{ vars.APPLE_TEAM_ID }}
        run: |
          xcrun notarytool submit "dist/pySimpleMask.dmg" \
            --apple-id "$APPLE_ID" \
            --password "$APPLE_APP_PASSWORD" \
            --team-id "$APPLE_TEAM_ID" \
            --wait

          xcrun stapler staple "dist/pySimpleMask.dmg"
          xcrun stapler validate "dist/pySimpleMask.dmg"

      - name: Check Gatekeeper
        run: |
          spctl \
            --assess \
            --type open \
            --context context:primary-signature \
            --verbose=4 \
            "dist/pySimpleMask.dmg"

      - name: Rename DMG with version
        run: |
          TAG="${{ github.event_name == 'push' && github.ref_name || github.event.inputs.version_tag }}"
          mv "dist/pySimpleMask.dmg" "dist/pySimpleMask-${TAG}-macos.dmg"

      - name: Upload macOS dmg artifact
        uses: actions/upload-artifact@v7
        with:
          name: macos-dmg
          path: dist/pySimpleMask-*-macos.dmg
          retention-days: 14

      - name: Delete temporary keychain
        if: always()
        run: |
          if [ -n "${KEYCHAIN_PATH:-}" ]; then
            security delete-keychain "$KEYCHAIN_PATH" || true
          fi
```

Note: the `TAG="${{ github.event_name == 'push' && ... }}"` expression is copied
verbatim from the existing `windows-build`/`linux-appimage-build` jobs for consistency
— on a manual `workflow_dispatch` run (no `version_tag` input is declared on this
workflow) it evaluates to an empty string, same pre-existing behavior as the other two
platform jobs. Not introduced by this task; not this task's job to fix.

- [ ] **Step 5: Update the `github-release` job**

In the `github-release` job, change:

```yaml
  github-release:
    name: Publish GitHub Release
    needs: [windows-build, linux-appimage-build]
```

to:

```yaml
  github-release:
    name: Publish GitHub Release
    needs: [windows-build, linux-appimage-build, macos-build]
```

Add a download step alongside the existing two (after "Download Linux AppImage
artifact"):

```yaml
      - name: Download macOS dmg artifact
        uses: actions/download-artifact@v8
        with:
          name: macos-dmg
          path: dist/
```

And extend the release step's `files:` list:

```yaml
      - name: Create/update GitHub Release
        uses: softprops/action-gh-release@v3
        with:
          tag_name: ${{ github.ref_name }}
          name: "pySimpleMask ${{ github.ref_name }}"
          generate_release_notes: true
          files: |
            dist/pySimpleMask-*-windows.exe
            dist/pySimpleMask-*-x86_64.AppImage
            dist/pySimpleMask-*-macos.dmg
```

- [ ] **Step 6: Run tests to verify they pass**

Run:
```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pip install -e ".[dev]"
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/packaging/test_build_releases_workflow.py -v
```
Expected: all PASS.

- [ ] **Step 7: Run the full test suite and lint**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests -q
/local/MQICHU/envs/l2606_simplemask_refact/bin/ruff check src tests
```

Expected: both PASS (all tests including the new `tests/packaging/` ones; ruff clean —
`.github/workflows/*.yml` and `pyproject.toml` aren't Python files ruff lints, so
nothing new for it to check there, but the new `tests/packaging/*.py` files must pass).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .github/workflows/build-releases.yml \
        tests/packaging/test_build_releases_workflow.py
git commit -m "feat(ci): build, sign, and notarize a macOS .dmg release artifact"
```

---

## After This Plan Is Executed

Real verification of the signing/notarization pipeline can't happen in this session —
there's no macOS runner and no access to the Apple credentials. Once all four tasks are
committed and pushed:

1. Go to **GitHub → Actions → Build GUI releases → Run workflow** and dispatch it
   manually from `master`.
2. Confirm the `macos-build` job's "Import Developer ID certificate" step logs
   `Developer ID Application: UChicago Argonne LLC (S5796PM55D)`.
3. Confirm "Notarize and staple DMG" reports `status: Accepted`.
4. Confirm the job completes and uploads a `macos-dmg` artifact — and that
   `github-release` did **not** run (manual dispatch, not a tag push), so nothing was
   published.
5. Once that manual run is green, the same workflow will include the macOS `.dmg` in
   the next `v*` tag push automatically — no further changes needed.

If notarization or Gatekeeper fails on that first real run, the most likely fix is
adjusting `packaging/macos/entitlements.plist`, not restructuring the workflow (see
"Open risk" in the design spec).
