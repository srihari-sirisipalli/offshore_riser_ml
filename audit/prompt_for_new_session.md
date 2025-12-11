Hello! I'm resuming work on the `offshore_riser_ml` project. My previous session ended after addressing an `IndentationError` and a `ModuleNotFoundError`. I also fixed a `RuntimeWarning` from `seaborn` in `modules/visualization/rfe_visualizer.py` and modified `modules/data_integrity/data_integrity_tracker.py` to prevent a `KeyError` during data tracking.

However, after these changes, some tests are still failing.

**Current Failing Tests:**
1.  `tests/ensembling_engine/test_ensembling_engine.py::TestEnsemblingEngine::test_mismatched_length_error`
2.  `tests/ensembling_engine/test_ensembling_engine.py::TestEnsemblingEngine::test_mismatched_index_error`
3.  `tests/rfe/test_rfe_controller.py::TestRFEController::test_full_loop_execution_logic`
4.  `tests/rfe/test_rfe_controller.py::TestRFEController::test_resume_logic`

**Tasks for this session:**

**Phase 1: Fix Remaining Failing Tests**
-   **Ensembling Engine Tests:** Modify `tests/ensembling_engine/test_ensembling_engine.py`. The `test_mismatched_length_error` and `test_mismatched_index_error` currently expect a `ValueError`. Due to the `@handle_engine_errors` decorator, these exceptions are now wrapped in `RiserMLException`. Update these tests to expect `RiserMLException` instead of `ValueError`.
-   **RFE Controller Tests:** Modify `tests/rfe/test_rfe_controller.py`. The `test_full_loop_execution_logic` and `test_resume_logic` tests are encountering `KeyError` within the `DataIntegrityTracker`. I have already implemented a fix in `modules/data_integrity/data_integrity_tracker.py` to prevent this `KeyError`. Please run the tests again. If these tests still fail, determine the new exception type (if any) and update the `pytest.raises` assertions accordingly to expect `RiserMLException`.
-   After applying these fixes, run the entire test suite (`python -m pytest ./tests/`) to ensure all tests pass and no new issues have been introduced.

**Phase 2: Implement Audit Recommendations and Enhancements**
The primary goal is to systematically go through the audit recommendations and implement the remaining enhancements, issues, and fixes as detailed in the `audit/*.md` files.

**Process for Audit Recommendations:**
1.  **Read and Consolidate:** Read through all files in the `audit/` directory:
    -   `audit/00_INDEX.md`
    -   `audit/01_CODE_QUALITY_AND_ARCHITECTURE.md`
    -   `audit/02_PERFORMANCE_OPTIMIZATION.md`
    -   `audit/03_MISSING_ANALYSES_AND_REPORTS.md`
    -   `audit/04_ADVANCED_VISUALIZATIONS.md`
    -   `audit/05_DEVELOPMENT_ROADMAP.md`
2.  **Extract Actionable Items:** From these documents, identify specific, actionable recommendations, issues, and enhancements that still need to be addressed. Pay close attention to items marked as high priority or P1.
3.  **Prioritize and Plan:** Create a new detailed TODO list using the `write_todos` tool, categorizing and prioritizing these items.
4.  **Implement Iteratively:** Implement each item from the prioritized TODO list. For each implementation:
    -   Adhere strictly to existing project conventions (code style, naming, architecture).
    -   Write or update unit tests to cover new functionality or bug fixes.
    -   Run relevant tests and project-specific build/linting/type-checking commands (e.g., `ruff check .`, `mypy .`) to ensure code quality and prevent regressions. If these commands are not explicitly known, please ask.
    -   Update the TODO list status as items are `in_progress`, `completed`, or `cancelled`.

**Overall Goal:**
The ultimate goal is to enhance the ML pipeline's quality, performance, and functionality based on the comprehensive audit, ensuring all tests pass and the codebase is robust and maintainable.

Please proceed with these tasks, starting with the test fixes.
