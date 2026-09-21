def test_the_per_question_table_reads_the_suite_the_gates_read():
    """Two true numbers for one quantity, three sections apart, neither labelled.

    The breakdown used to take the first result carrying one, which is the
    *uncalibrated* run, while the gates are computed from the calibrated one.
    On Banking77 seed 2 the gate reported `intent` at 0.7126 and this table
    reported 0.7194. Nobody noticed for as long as calibration never changed a
    decision: an isotonic map is monotone per class and not jointly, so it can
    reorder two classes and move the argmax, and on a 77-way choice it did.
    """
    from trigon.calibration.metrics import CalibrationReport
    from trigon.evals.harness import QuestionAccuracy, SuiteResult
    from trigon.evals.report import render_markdown

    def suite(name: str, accuracy: float) -> SuiteResult:
        return SuiteResult(
            suite=name,
            model="m",
            n_cases=10,
            n_questions=10,
            accuracy=accuracy,
            baseline_accuracy=0.1,
            calibration=CalibrationReport(
                n=10,
                ece=0.02,
                adaptive_ece=0.02,
                mce=0.05,
                brier=0.3,
                nll=0.5,
                accuracy=accuracy,
                mean_confidence=accuracy,
                bins=(),
            ),
            latency_p50_ms=1.0,
            latency_p99_ms=2.0,
            mean_prefill_tokens=100,
            per_question={
                "intent": QuestionAccuracy(
                    question_id="intent", n=10, accuracy=accuracy, baseline_accuracy=0.1
                )
            },
        )

    before, after = suite("x/uncalibrated", 0.7194), suite("x/calibrated", 0.7126)
    text = render_markdown([before, after], gated=after)
    assert "0.7126" in text, "the table must show the accuracy the gates were computed from"
    assert "x/calibrated" in text, "and must say which run it came from"

    # Without `gated` the old behaviour stands, so single-suite callers such
    # as `trigon eval` are unaffected.
    assert "0.7194" in render_markdown([before])
