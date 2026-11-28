"""Built-in HUD layouts.

Importing a module here runs its ``register`` call, which is how
``framework.ui.layout`` finds them without a hard-coded table.
"""

from framework.ui.layouts import desktop

__all__ = ["desktop"]
