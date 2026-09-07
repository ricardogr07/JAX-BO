# RepoSage Standards Audit

- Root path: `C:\git\JAX-BO`
- Profile: data science / ML (3 training, 1 serving file(s))
- Uncertain checks: 2

**Grade: 3/6**

## Standard 0: Reproducible - PASS (3/3)

| Check | Status | Evidence | Remediation |
| --- | --- | --- | --- |
| Environment spec | PASS | pyproject.toml covers 5 confident third-party import(s) |  |
| Dependency pinning | PASS | uv.lock present and newer than or equal to pyproject.toml |  |
| Determinism | PASS | no random sources detected in model code |  |

## Standard 1: Legible - PASS (3/3)

| Check | Status | Evidence | Remediation |
| --- | --- | --- | --- |
| Version control | PASS | 236 commits with legible subjects |  |
| Documentation | PASS | docstring coverage 70% (76/108) |  |
| Logging | PASS | no stray print() calls in checked modules |  |

## Standard 2: Structured - FAIL (2/3)

| Check | Status | Evidence | Remediation |
| --- | --- | --- | --- |
| Package | UNCERTAIN | looks installable; rerun with --run-subprocess-checks to verify | Verify an editable install in a clean environment. |
| Module boundaries | PASS | no raw I/O in model/serving code |  |
| Config externalization | PASS | no credential-shaped literals found |  |

## Standard 3: Proven - FAIL (2/3)

| Check | Status | Evidence | Remediation |
| --- | --- | --- | --- |
| Test suite | UNCERTAIN | 18 test file(s) present; rerun with --run-subprocess-checks to execute them | Rerun with --run-subprocess-checks to execute the suite. |
| Behavioral coverage | PASS | 45/79 test functions make value-bearing assertions (ratio 0.57) |  |
| Evaluation gate | PASS | evaluation gate jaxbo/test_functions.py runs in CI or as a test |  |

## Standard 4: Shipped - PASS (3/3)

| Check | Status | Evidence | Remediation |
| --- | --- | --- | --- |
| Deploy independence | PASS | deploy/publish workflow: .github/workflows/ci.yml |  |
| Environment isolation | PASS | Dockerfile copies a lockfile and installs from it |  |
| CI/CD | PASS | deploy job 'test' in .github/workflows/ci.yml needs the test job |  |

## Standard 5: Accountable - FAIL (0/3)

| Check | Status | Evidence | Remediation |
| --- | --- | --- | --- |
| Queryable logs | FAIL | no log exporter in serving code: logs die with the process | Ship logs to a queryable backend (OpenTelemetry, Azure Monitor, or structlog handler). |
| Metric tracking | FAIL | serving surface emits no production metrics | Instrument the missing surface: production metrics for serving, experiment tracking for training. |
| Alerting | FAIL | no alert rules found (Prometheus rules, Azure metric alerts, or a scheduled check) | Add an alert rule that fires when an emitted metric crosses a threshold. |

## Fix list

1. Standard 2 (s2.package): Verify an editable install in a clean environment.
2. Standard 3 (s3.suite): Rerun with --run-subprocess-checks to execute the suite.
   Priority: Standard 3, Proven, carries the highest weight; nothing above it can be trusted until it passes
3. Standard 5 (s5.logs): Ship logs to a queryable backend (OpenTelemetry, Azure Monitor, or structlog handler).
4. Standard 5 (s5.metrics): Instrument the missing surface: production metrics for serving, experiment tracking for training.
5. Standard 5 (s5.alerting): Add an alert rule that fires when an emitted metric crosses a threshold.
