"""
Utility package setup.

Enables pandas Copy-on-Write globally to cut down on unnecessary DataFrame duplication
while keeping mutation safety.
"""

import pandas as pd

# Reduce implicit copies across the pipeline.
pd.options.mode.copy_on_write = True
