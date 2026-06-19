# MNEMOS SBOM Path

MNEMOS does not yet publish a release SBOM. This directory records the intended
SBOM workflow so dependency transparency has an explicit home before external
release packaging.

## Target Artifacts

Generate at least one of the following for every external release candidate:

- container image SBOM for the MNEMOS service image;
- Python environment SBOM from the resolved package set;
- combined release SBOM covering service, sidecars, and benchmark/runtime
  containers.

## Suggested Tools

Use whichever tool is already approved in the release environment:

```bash
syft packages dir:. -o spdx-json > docs/sbom/mnemos-source.spdx.json
syft packages docker:mnemos-service:<release-tag> -o spdx-json > docs/sbom/mnemos-image.spdx.json
```

If `syft` is not available, use an equivalent SPDX or CycloneDX generator and
record the command in the release notes.

## Release Rule

Before publishing an external release:

- generate the SBOM from a clean checkout or release image;
- record tool name and version;
- record the Git commit and image digest;
- store the generated artifact outside normal benchmark output directories;
- link the artifact from `docs/dependency_map.md`.
