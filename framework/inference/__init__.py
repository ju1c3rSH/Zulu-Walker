"""Platform-agnostic, task-agnostic inference kernel.

The package is split into three independent axes:

* engines: adapters around third-party inference runtimes
* manifest: the run-time contract and the offline build recipe
* wrappers: task adaptation layer producing capability objects

Entry points live in the submodules: contract for the data structures and
the engine protocol, manifest for parsing, ops for the generic
primitives, registry for engine selection, runtime and facade for the
lifecycle, and wrappers plus engines for the task and backend adapters.
"""
