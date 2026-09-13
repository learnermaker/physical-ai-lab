# Security Policy

## Supported versions

This is a workshop repository. Only the latest commit on `master` is maintained.

## Reporting a vulnerability

This repo does not run as a service — it runs locally on participants' machines. The attack surface is limited to:

- Dependencies listed in `requirements.txt`
- The local FastAPI hub server (binds to `localhost:8000` only, not exposed to the network)

If you find a genuine security issue (e.g. a dependency with a known CVE, an unsafe code pattern, or a way the local server could be exploited), please **do not open a public issue**.

Instead, report it via [GitHub's private vulnerability reporting](https://github.com/learnermaker/physical-ai-lab/security/advisories/new).

Include:
- A description of the issue
- Which file or dependency is affected
- Steps to reproduce, if applicable

You can expect a response within 7 days. If the issue is confirmed, a fix will be released and credited to you in the changelog unless you prefer to remain anonymous.
