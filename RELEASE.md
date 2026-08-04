# nvector Release Workflow

nvector uses **PDM** for package management and versioning.

Releases are created from Git tags. A package is only published to PyPI when a version tag beginning with `v` is pushed and all CI checks pass.

---

# Workflows

The repository uses two GitHub Actions workflows:

| Workflow | Purpose | Trigger |
|----------|----------|----------|
| `ci.yml` | Run tests and validate builds | Push, Pull Request, Schedule |
| `release.yml` | Build and publish package to PyPI | Version tag (`v*`) |

---

# Branch Strategy

The nvector repository follows a Git Flow inspired workflow.

Development work is performed on:

```text
develop
```

Feature branches should be merged into `develop`.

When a release is ready:

1. Update `CHANGELOG.md`.
2. Bump the version using `pdm bump`.
3. Merge `develop` into `master`.
4. Push the updated `master` branch.
5. Create and push a release tag.
6. GitHub Actions publishes the tagged version to PyPI.

---

# 1. Development

Run tests locally before pushing changes:

```bash
pdm sync -dG test
pdm run pytest
```

Optional quality checks:

```bash
pdm run ruff check .
pdm run mypy ./src
```

Commit and push normally:

```bash
git add .
git commit -m "fix(core): correct edge case in geodesic calculation"
git push
```

Every push and pull request automatically triggers CI.

---

# 2. Continuous Integration

The CI workflow runs automatically on:

- Pushes to the configured integration branches.
- Pull requests targeting the integration branches.
- Scheduled weekday runs.

The workflow tests:

- Python 3.10
- Python 3.11
- Python 3.12
- Python 3.13
- Python 3.14
- Python 3.15 (allowed failure)

CI performs:

```bash
pdm install -dG test
pdm run pytest
```

and validates package creation:

```bash
pdm build
```

It also verifies that the generated wheel can be installed:

```bash
pip install dist/*.whl
```

A release cannot be published unless all required CI jobs pass.

---

# 3. Update the Changelog

Before creating a release, update `CHANGELOG.md`.

Generate commits since the previous version:

```bash
git log vX.Y.Z..HEAD --oneline > log.txt
```

For example:

```bash
git log v1.0.6..HEAD --oneline > log.txt
```

Review the changes and summarize them into user-facing release notes.

The project uses **git-cliff** to help generate changelog entries:

```bash
pdm run append-changelog
```

Review and edit the generated content before committing.

Commit the updated changelog:

```bash
git add CHANGELOG.md
git commit -m "docs: update changelog for vX.Y.Z"
git push
```

---

# 4. Determine the Next Version

nvector uses **pdm-bump** for version management.

View the suggested next version:

```bash
pdm bump suggest
```

Common version bumps:

```bash
pdm bump micro
```

```bash
pdm bump minor
```

```bash
pdm bump major
```

For pre-releases:

```bash
pdm bump pre-release --pre alpha
```

```bash
pdm bump pre-release --pre beta
```

```bash
pdm bump pre-release --pre release-candidate
```

After updating the version:

```bash
git add .
git commit -m "chore(release): bump version"
git push
```

Create the release tag:

```bash
pdm bump tag
```

---

# 5. Create a Release Tag

Create the version tag:

```bash
git tag vX.Y.Z
```

Example:

```bash
git tag v1.0.7
```

Verify:

```bash
git tag
```

Push the tag:

```bash
git push origin v1.0.7
```

or

```bash
git push --tags
```

---

# 6. Publish to PyPI

Publishing is automatic.

When a tag matching:

```text
v*
```

is pushed:

```text
v1.0.7
v2.0.0
v1.1.0rc1
```

GitHub Actions:

1. Builds the package using PDM.
2. Verifies the build artifacts.
3. Verifies the version does not already exist on PyPI.
4. Publishes using GitHub OIDC trusted publishing.
5. Uploads the source distribution and wheel to PyPI.

No manual PyPI upload is required.

---

# 7. Verify the Release

Install from PyPI:

```bash
pip install -U nvector
```

Verify the installed version:

```python
import nvector as nv

print(nv.__version__)
```

Run a quick functional check:

```python
import nvector as nv

point = nv.GeoPoint.from_degrees(latitude=60.0, longitude=5.0)
print(point)
```

Optionally run the test suite against the installed package:

```bash
pytest --pyargs nvector
```

---

# Commit Message Conventions

The repository uses conventional commits and git-cliff.

Preferred commit types:

```text
feat:
fix:
docs:
refactor:
perf:
test:
style:
chore:
ci:
```

Examples:

```text
feat(core): add geodesic helper
```

```text
fix(objects): correct distance calculation
```

```text
docs: update release instructions
```

```text
ci: update GitHub Actions workflows
```

These commit messages are used when generating changelogs.

---

# Release Checklist

Before release:

- [ ] All tests pass locally.
- [ ] CI passes on GitHub.
- [ ] CHANGELOG.md updated.
- [ ] Version bumped using `pdm bump`.
- [ ] Changes committed and pushed to `develop`.
- [ ] `develop` merged into `master`.
- [ ] Updated `master` pushed.
- [ ] Release tag created.

Release:

- [ ] Push tag `vX.Y.Z`.

After release:

- [ ] Verify PyPI package exists.
- [ ] Verify installation from PyPI.
- [ ] Verify reported version.
- [ ] Verify basic functionality.

---

# Workflow Summary

```text
feature branch
      │
      ▼
   develop
      │
      ▼
    ci.yml
      │
      ▼
Update CHANGELOG.md
      │
      ▼
   pdm bump
      │
      ▼
Merge develop -> master
      │
      ▼
git push origin master
      │
      ▼
git tag vX.Y.Z
      │
      ▼
git push origin vX.Y.Z
      │
      ▼
 release.yml
      │
      ▼
 Build package
      │
      ▼
 Verify artifacts
      │
      ▼
 Upload to PyPI
      │
      ▼
pip install nvector
```


