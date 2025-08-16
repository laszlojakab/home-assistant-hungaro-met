"""Module of HungaroMet integration."""

from dataclasses import dataclass

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .const import LOGGER
from .coordinator import (
    HungaroMet10MinDataUpdateCoordinator,
    HungaroMetMsgCloudTypeDataUpdateCoordinator,
)


@dataclass
class HungaroMetWeatherRuntimeData:
    """The runtime data for the HungaroMet weather integration."""

    coordinator_10_min: HungaroMet10MinDataUpdateCoordinator
    coordinator_msg_cloud_type: HungaroMetMsgCloudTypeDataUpdateCoordinator


type HungaroMetConfigEntry = ConfigEntry[HungaroMetWeatherRuntimeData]
"""The config entry for the HungaroMet weather integration."""

PLATFORMS = [Platform.WEATHER, Platform.SENSOR]
"""The platforms supported by the HungaroMet weather integration."""


async def async_setup_entry(
    hass: HomeAssistant, config_entry: HungaroMetConfigEntry
) -> bool:
    """
    Initializes the predictor sensor based on the config entry.

    Args:
        hass: The Home Assistant instance.
        config_entry: The config entry which contains information gathered by the config flow.

    Returns:
        The value indicates whether the setup succeeded.

    """
    coordinator_10_min = HungaroMet10MinDataUpdateCoordinator(
        hass, config_entry, LOGGER
    )
    await coordinator_10_min.async_config_entry_first_refresh()

    coordinator_msg_cloud_type = HungaroMetMsgCloudTypeDataUpdateCoordinator(
        hass, config_entry, LOGGER
    )
    await coordinator_msg_cloud_type.async_config_entry_first_refresh()

    config_entry.runtime_data = HungaroMetWeatherRuntimeData(
        coordinator_10_min=coordinator_10_min,
        coordinator_msg_cloud_type=coordinator_msg_cloud_type,
    )

    await hass.config_entries.async_forward_entry_setups(config_entry, PLATFORMS)

    return True


async def async_unload_entry(
    hass: HomeAssistant, config_entry: HungaroMetConfigEntry
) -> bool:
    """
    Executed when a config entry is unloaded by Home Assistant.

    Args:
        hass: The Home Assistant instance.
        config_entry: The config entry being unloaded.

    Returns:
        The value indicates whether the unloading succeeded.

    """
    for platform in PLATFORMS:
        await hass.config_entries.async_forward_entry_unload(config_entry, platform)

    return True
