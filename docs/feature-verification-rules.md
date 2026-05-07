\# Feature Verification Rule



Whenever implementing a new feature, behavioral change, persistence field, UI message, state, or pipeline logic:



Do not stop at “tests pass”.  

Perform systematic feature verification across all affected paths.



\---



\## 1. State Coverage



Identify all relevant states, branches, variants, and execution paths affected by the change.



Examples:

\- completed

\- failed

\- partial\_failed

\- insufficient\_quality

\- empty results

\- fallback/default paths

\- retry paths

\- validation failures



Verify every affected state manually or through tests.



Do not verify only the happy path.



\---



\## 2. Persistence Verification



If data should be saved:



\- verify it exists in the database

\- verify correct values are persisted

\- verify update/save paths actually include the field

\- verify `refresh\_from\_db()` behavior when relevant

\- verify migrations/default values behave correctly



Do not assume assigned values are persisted automatically.



\---



\## 3. UI Rendering Verification



If data is user-facing:



\- verify it appears in the UI

\- verify the correct source of truth is rendered

\- verify legacy/fallback text is not accidentally overriding it

\- verify empty/null behavior

\- verify placement and visibility in real UI flows



Do not assume persisted data is actually shown correctly.



\---



\## 4. Source-of-Truth Consistency



Check whether:



\- multiple layers generate the same message/data independently

\- legacy logic still exists

\- UI/template/view/pipeline are using the same canonical field/helper

\- duplicated constants/messages exist



Prefer one canonical source of truth for:

\- user-facing messages

\- statuses

\- configuration values

\- thresholds

\- derived summaries



\---



\## 5. Backward Compatibility



Verify:



\- existing flows still work

\- old records do not break

\- fallback/default behavior still works

\- optional fields behave correctly

\- old UI paths still render safely



Do not silently break existing runs, records, or topics.



\---



\## 6. Terminal Path Audit



For pipeline/state-machine/workflow systems:



Explicitly audit all terminal paths after every behavioral change.



Examples:

\- completed

\- failed

\- partial\_failed

\- skipped

\- insufficient\_quality

\- validation\_failed



Do not assume one fixed path represents the whole system.



\---



\## 7. Manual Real-World Run



At least one real run through the actual UI/entrypoint is required for:



\- pipeline changes

\- persistence changes

\- state changes

\- user-facing messaging changes

\- source handling changes

\- workflow behavior changes



Do not rely only on unit tests.



\---



\## 8. Tests



Tests should verify:



\- behavior

\- persistence

\- rendering/output

\- edge/fallback cases

\- terminal states

\- regression safety



Do not only test returned values.  

Test persisted and rendered behavior too.



Use `refresh\_from\_db()` where relevant.



\---



\## 9. Observability Verification



If the feature affects system behavior:



\- verify logs/metrics/debug information still make sense

\- verify failures remain diagnosable

\- verify debugging became easier, not harder

\- verify snapshots/traces stay consistent

\- verify user-facing summaries match technical diagnostics



Observability is part of product quality.



\---



\## 10. Commit Boundary



Only commit after:



\- tests pass

\- manual verification passes

\- all affected states/paths were checked

\- no inconsistent legacy behavior remains

\- persistence and UI were verified

\- observability still works



One commit should represent one coherent, verified change.

