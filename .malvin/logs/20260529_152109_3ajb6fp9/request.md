Satisfy all constraints.


Scope Constraints:
- Just get quality gates to pass.

General Constraints:
- All quality gates (see below) to pass
- No serious bugs in scope
- No serious time-complexity inefficiencies in scope
- No serious memory-complexity inefficiencies in scope
- Each unit test tests something meanignful. Simple tests are fine. Bogus tests are not.
- Any code you write should be idiomatic. (For example: Don't use ".inc" files in Rust.)
- When you code, stay in scope.

If you write code:
- Write real unit tests, even if it seem like you have to write a lot of them. Do your best. Don't use tricks or make superficial unit tests just to pass coverage gates.


Quality Gates:

- `kiss check`
- `ruff check .`
- `cd rust && cargo clippy --all-targets --all-features -- -D warnings -W clippy::cargo`
- `make test`


Latest & up-to-date quality gate run output is in: `./.malvin/logs/20260529_152109_3ajb6fp9/quality_gates.log`.