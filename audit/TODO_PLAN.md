## Audit TODO (High Priority)

- [x] Parquet migration for internal artifacts (replace Excel where not user-facing; ensure readers updated and tests adjusted)
- [x] DataFrame copy audit and enable pandas copy-on-write
- [x] Add caching layer for repeatable computations (joblib.Memory + lru_cache)
- [ ] Harden resume/round directory creation (atomic writes, completeness markers)
- [ ] HPO grid/resource limits (max configs/time/memory validation)
- [ ] Standardize error handling with RiserMLException hierarchy
- [ ] Expand configuration validation bounds and resource guardrails
- [ ] Introduce BaseEngine abstraction and dependency injection
- [ ] Add streaming/chunked data loading support
- [ ] Implement Phase-1 analyses (data integrity, statistical significance, safety thresholds, reproducibility)
- [ ] Add structured logging and quality checks log
- [ ] Deployment readiness and integration test tracking
