# RepoSage Standards Audit

- Root path: `C:\git\JAX-BO`
- Profile: data science / ML (3 training, 1 serving file(s))
- Uncertain checks: 0

**Grade: 2/6**

## Standard 0: Reproducible - FAIL (1/3)

| Check | Status | Evidence | Remediation |
| --- | --- | --- | --- |
| Environment spec | PASS | pyproject.toml covers 5 confident third-party import(s) |  |
| Dependency pinning | FAIL | uv.lock predates the last change to pyproject.toml | Regenerate the lockfile after changing the spec. |
| Determinism | FAIL | benchmarks/bench_train.py:35 random. without seed | Set an explicit seed (random_state=, np.random.seed, torch.manual_seed). |

## Standard 1: Legible - FAIL (1/3)

| Check | Status | Evidence | Remediation |
| --- | --- | --- | --- |
| Version control | PASS | 235 commits with legible subjects |  |
| Documentation | FAIL | docstring coverage 59% below 70% | Add docstrings to public functions and classes. |
| Logging | FAIL | print() used without a logging framework: benchmarks/profile_train.py (1) | Replace print() with the logging/structlog module. |

## Standard 2: Structured - FAIL (2/3)

| Check | Status | Evidence | Remediation |
| --- | --- | --- | --- |
| Package | FAIL | pip install -e failed: or originates from a subprocess, and is likely not a problem with pip.
error: subprocess-exited-with-error

pip subprocess to install build dependencies did not run successfully.
exit code: 1

See above for output.

note: This error originates from a subprocess, and is likely not a problem with pip. | Fix the packaging so an editable install succeeds. |
| Module boundaries | PASS | no raw I/O in model/serving code |  |
| Config externalization | PASS | no credential-shaped literals found |  |

## Standard 3: Proven - PASS (3/3)

| Check | Status | Evidence | Remediation |
| --- | --- | --- | --- |
| Test suite | PASS | pytest passed; 98 tests collected |  |
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

1. Standard 0 (s0.lockfile): Regenerate the lockfile after changing the spec.
2. Standard 0 (s0.determinism): Set an explicit seed (random_state=, np.random.seed, torch.manual_seed).
3. Standard 1 (s1.docs): Add docstrings to public functions and classes.
4. Standard 1 (s1.logging): Replace print() with the logging/structlog module.
5. Standard 2 (s2.package): Fix the packaging so an editable install succeeds.
6. Standard 5 (s5.logs): Ship logs to a queryable backend (OpenTelemetry, Azure Monitor, or structlog handler).
7. Standard 5 (s5.metrics): Instrument the missing surface: production metrics for serving, experiment tracking for training.
8. Standard 5 (s5.alerting): Add an alert rule that fires when an emitted metric crosses a threshold.
