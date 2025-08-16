"""Module of constants."""

import logging
from datetime import timedelta
from typing import Final, Literal

DOMAIN: Final = "hungaromet"
"""The domain of the HungaroMet integration."""

UPDATE_INTERVAL: Final = timedelta(minutes=3)
"""The interval at which to update data from the HungaroMet Server."""

LOGGER: Final = logging.getLogger(__package__)
"""The logger for the HungaroMet integration."""

CONF_CLIMATE: Final = "climate"
"""The climate configuration key."""

type CLIMATE_TYPE = Literal["humid", "sub-humid", "semi-arid", "arid"]
"""The climate types supported by the HungaroMet integration."""
