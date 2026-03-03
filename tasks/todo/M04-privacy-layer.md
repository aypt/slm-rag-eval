# M04 · Privacy layer (Presidio PII sanitization)

Context: repo slm-rag-eval; M01–M03 done. This file is the complete spec; also
follow AGENTS.md. This task implements the project's core differentiator: sensitive
retrieved contexts must never reach an external judge in raw form.

Task: implement src/slm_rag_eval/privacy/ using Microsoft Presidio.

Requirements:
1. Add runtime deps: presidio-analyzer, presidio-anonymizer, spacy. Extend
   `make setup` to also run `python -m spacy download en_core_web_lg`, and document
   in README why the model is needed.
2. Sanitizer.sanitize(texts: list[str]) -> SanitizedBatch{texts: list[str],
   mapping: dict[str, str]}. Detect entities (configurable list; default PERSON,
   EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, IP_ADDRESS, LOCATION, US_SSN;
   configurable score threshold, default 0.4). Replace each unique surface form
   with a stable placeholder like <PERSON_1>, consistent across all texts in the
   batch (question/answer/contexts). mapping is placeholder -> original.
3. restore(text: str, mapping) -> str to de-anonymize judge output for display.
4. Wire into metrics/registry.evaluate honoring Settings.privacy_mode
   ("off" | "mask", default "mask"): when masking, sanitize the EvalRequest before
   ANY judge call. The mapping stays in-memory only — never logged, never persisted.
   Keep Sanitizer construction lazy/injectable so unit tests that don't test privacy
   can pass a no-op sanitizer and stay fast.
5. Privacy invariant test (name it test_privacy_invariant_no_pii_leaves_process):
   with privacy_mode="mask" and fixtures containing invented PII (fake names,
   emails, phone numbers only — never real data), assert that no raw PII substring
   appears in any prompt captured by FakeLLMClient.calls. Mark clearly in a comment
   that this test must never be weakened (AGENTS.md rule 2).
6. Presidio-dependent tests may be skipped with a clear reason if the spaCy model
   is unavailable in the environment, but the invariant test must run using a
   stubbed detector in that case so the wiring is always verified.

Definition of done: `make check` green; README "Privacy layer" section explains the
threat model, what is and is not protected (e.g., NER misses are possible), and the
entity/threshold configuration; PROGRESS.md updated.
