# MIMIC-III

MIMIC-III requires credentialed access and is not redistributed in this
repository. Request access to the processed
[MIMIC-III-Ext-tPatchGNN](https://physionet.org/content/mimic-iii-ext-tpatchgnn/1.0.0/)
benchmark on PhysioNet.

Place the resulting file at:

```text
data/mimic/processed/mimic.pt
```

The official file contains patient records represented as
`(patient_id, times, values, mask)`, where `values` and `mask` have shape
`[T, 96]`.
