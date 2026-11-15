# Ethics and privacy

- **No real student records or transcripts in this repo.** The .gitignore blocks common data formats.
- **Synthetic data is labeled.** Metrics from synthetic data demonstrate the pipeline only.
- **Ecological fallacy.** Tract-level indicators describe neighborhoods, not individual students.
- **Association, not causation.** SHAP explains model predictions, not causes of absenteeism.
- **Framing.** Outputs are "support priority" or "offer of support," never labels about a child.
- **Human in the loop.** Counselors see supporting evidence and decide; the model never acts alone.
- **Fairness audit.** Report error rates and calibration by race/ethnicity, income, disability, and English-learner status.
- **Text risks.** Check theme detection across dialects and writing registers. Run text models locally. Consider consent and the chilling effect of scoring interviews.
- **Approvals.** Obtain IRB review and a data-use agreement before using real data.
