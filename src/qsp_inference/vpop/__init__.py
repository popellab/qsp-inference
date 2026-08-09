"""Virtual-population inference: ``docs/model-draft.tex`` and nothing else.

The fit is :mod:`~qsp_inference.vpop.fit`, driven by pdac-build's
``workflows/vpop_fit.py``. Nothing is re-exported here: every module needs jax
and numpyro, and a package-level import would load an optional dependency on
every ``import qsp_inference.vpop``. Import the modules directly.

Six of them, in the order the chain runs:

``mechanism``  the corpus and the surrogate to a ``Mechanism``: eq:mech, eq:readout
``predict``    ``tau_B(phi)``: eq:crn through eq:stat
``rows``       what a source printed, and the model's expectation of it: eq:smoothq
``blocks``     who is drawn with whom, and ``V_B``: eq:V, eq:Ec, eq:Vsplit
``fit``        eq:pop through eq:post as a numpyro model
``reports``    the metric, and what the rows determine: eq:ginfo, eq:zcost, eq:ratio

The fixed-cloud route that used to share this namespace is in
:mod:`qsp_inference.legacy`.
"""
