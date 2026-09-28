"""The compatibility route's path, importable without the server's dependencies.

The site build (`scripts/build_site.py`, the `site` extra) documents this route
and runs without FastAPI. It read the path from `trigon.server.compat`, which
imports FastAPI at module level, so the first Pages build on a runner died on
`No module named 'fastapi'` while every test here passed with the server extra
installed. A constant belongs where the fewest things have to import to read it.
"""

# The incumbent's endpoint path, exactly as their published contract states it.
# An interoperability detail, not a name this project uses: a migration changes
# a base URL and nothing else only if the path is theirs. It is written here
# once, and `tests/test_no_incumbent_names.py` allows it here and nowhere else
# outside the generated spec and the tests that exercise it.
COMPAT_PATH = "/v1/systemone"
