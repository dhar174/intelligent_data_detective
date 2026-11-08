# Separate finding: heatmap renderer references undefined cols

This is a pre-existing finding, not repaired by the figure-provenance change.
Inspected baseline: `ffd44169e49ddb11158bef99104334e2fb792a36`.

The actual generated Cell 32 `create_correlation_heatmap` can fail for a valid registered numeric DataFrame with `columns=["gross", "net"]`:

```text
Failed to generate correlation heatmap: name 'cols' is not defined
```

Its save-name construction references `cols`, which is not defined locally by the function. An isolated offline namespace reproduces the error. This is separate from the live missing-scatter metadata mismatch: the failing live worker used custom Python and `create_scatter_plot`.

The heatmap function's AST is unchanged from baseline in this repair. It was excluded from the four successful plotting-tool path-delivery regressions rather than making a fixture supply an accidental global `cols` or claiming heatmap success. No test permanently asserts that this bug must exist.

A separate approved follow-up should reproduce and repair the column selection/save-name binding, ensure the image is saved through the canonical current-run resolver, expose its authoritative path in model-visible content, and test actual saved output plus failure behavior. Do not weaken worker, join, evaluator, or packager integrity checks to accommodate its failure.

No live execution or additional provider spend was used to discover this finding. It remains a potential live-workflow risk to consider before further campaign authorization. No issue was opened or closed automatically; #165 remains open and #129 remains deferred.
