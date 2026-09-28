"""Analytics package for UrbanTransit IQ.

Modules in this package turn the integrated facts (``data/processed``) and the
feature frames (``data/features``) into decision-support analytics: route
metrics, stop metrics, demand/peak patterns, crowding/underutilisation,
bunching, anomalies, event impact, passenger segmentation, and frequency analysis.

The single entry point is :func:`src.analytics.run.run_all`, which computes
every analytic, validates the row counts it produced, writes the frames under
``data/analytics`` and mirrors them into the logical ``analytics`` HDFS area.
"""