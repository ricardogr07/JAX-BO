# Security Policy

## Supported versions

Only the latest release line receives security fixes.

| Version | Supported |
|---|---|
| latest release (see [PyPI](https://pypi.org/project/jaxbo/)) | yes |
| older releases | no |

## Reporting a vulnerability

Please do not report security issues through the public issue tracker.

Report privately instead, using either channel:

- **GitHub private vulnerability reporting** (preferred):
  [open a draft security advisory](https://github.com/ricardogr07/JAX-BO/security/advisories/new)
- **Email:** rgr.5882@gmail.com with "jaxbo security" in the subject

Include the affected version, a minimal reproduction, and the impact you
see. You can expect an acknowledgment within 7 days. Once a fix is
available it ships in a patch release and the advisory is published; you
will be credited unless you prefer otherwise.

## Scope notes

jaxbo is a numerical library: it executes no network I/O and evaluates no
untrusted code paths of its own. The most likely real-world issues are
supply-chain ones (a compromised or vulnerable dependency) or unsafe
deserialization of model parameters from untrusted sources. Reports in
those areas are very welcome.
