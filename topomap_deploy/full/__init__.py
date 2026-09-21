"""Full original-topomap integration framework; no runtime starts on import.

This is separate from the diagnostic marker navigator in topomap_deploy.runtime.
Concrete adapters live in observations/lifecycle/sim_io; bootstrap assembles them
only on explicit invocation. They have not yet been runtime-validated.
"""
