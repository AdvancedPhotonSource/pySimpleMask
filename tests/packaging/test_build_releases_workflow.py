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
