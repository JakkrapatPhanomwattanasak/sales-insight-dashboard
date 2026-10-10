# Understanding Sales Insight Dashboard

- Start at `data_app.create_app`; the data work lives in `pipeline.py`.
- Trace CSV upload → schema/value validation → deduplication → split → preprocessing → Ridge → metrics → JSON → dashboard.
- Explain why preprocessing fits on training rows only and how unknown categories are handled internally.
- Why MAE and a mean baseline? They make prediction error interpretable in the data's revenue units.
- Explain why a regression score on synthetic data cannot establish real business impact.
- Explain why random row splitting must change for a true time-series forecast.
- Explain report preservation when an upload fails and the shared in-memory model's limitations.
- The UI inserts uploaded labels using textContent rather than raw HTML.

Practice: run the demo, upload a malformed CSV, verify the old report stays available, make an estimate, download the JSON, and walk through one test.

Resume wording after you can reproduce and explain the implementation:
"Developed a Flask sales-analysis dashboard integrating CSV ingestion, data validation, preprocessing, regression evaluation, a prediction API, and downloadable reports."

This is an illustrative revenue estimator, not a deployed forecasting system.
