"""Module for HungaroMet integration coordinators."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Self

import pandas as pd
from homeassistant.const import (
    ATTR_LATITUDE,
    ATTR_LONGITUDE,
    CONF_RADIUS,
    CONF_ZONE,
)
from homeassistant.helpers.storage import Store
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from .api import CloudMask, HungaroMet10MinDataProvider, HungaroMetMsgCloudTypeApi
from .const import UPDATE_INTERVAL

if TYPE_CHECKING:
    import logging

    from homeassistant.core import HomeAssistant

    from . import HungaroMetConfigEntry


class HungaroMet10MinDataUpdateCoordinator(DataUpdateCoordinator[pd.DataFrame]):
    """Data update coordinator which loads the from HungaroMet 10 minute data provider."""

    def __init__(
        self: Self,
        hass: HomeAssistant,
        config_entry: HungaroMetConfigEntry,
        logger: logging.Logger,
    ) -> None:
        """
        Initializes a new instance of HungaroMet10MinDataUpdateCoordinator class.

        Args:
          hass: The Home Assistant instance.
          config_entry: The configuration entry for the integration.
          logger: The logger instance.
        """
        super().__init__(
            hass,
            logger,
            name="10 min data",
            update_interval=UPDATE_INTERVAL,
        )
        self._config_entry = config_entry
        self._api = HungaroMet10MinDataProvider()
        self._store = Store(
            hass, 1, f"{config_entry.domain}_{config_entry.entry_id}_10_min_data"
        )

    async def _async_update_data(self) -> dict[str, Any]:
        """Updates stored data from data provider."""
        if self.data is None:
            try:
                stored_data = await self._store.async_load()
                self.data = pd.DataFrame.from_dict(stored_data["data"])
                self.data.index = pd.to_datetime(stored_data["index"])
            except BaseException:
                self.logger.exception("Failed to load data from store")

        zone = self.hass.states.get(self._config_entry.data[CONF_ZONE])
        if not zone:
            return self.data

        latitude = zone.attributes.get(ATTR_LATITUDE)
        longitude = zone.attributes.get(ATTR_LONGITUDE)

        if latitude is None:
            return self.data

        if longitude is None:
            return self.data

        after_timestamp = None if self.data is None else self.data.index.max()

        data = await self._api.get_location_data(
            latitude,
            longitude,
            ["ta", "p", "fs", "fx", "fsd", "v", "r", "u", "suv", "sr"],
            max_radius=self._config_entry.data[CONF_RADIUS],
            after_timestamp=after_timestamp,
        )

        if data is None:
            return self.data

        if self.data is not None:
            data = pd.concat([self.data, data], axis="index")
            old_data_cutoff = pd.Timestamp.now(UTC).normalize() - pd.Timedelta(days=1)
            data = data[data.index > old_data_cutoff]

        try:
            await self._store.async_save(
                {
                    "data": data.to_dict(orient="list"),
                    "index": list(data.index),
                }
            )
        except BaseException:
            self.logger.exception("Failed to save data to store")

        return data


class HungaroMetMsgCloudTypeDataUpdateCoordinator(
    DataUpdateCoordinator[list[tuple[datetime, CloudMask]]]
):
    """Data update coordinator for HungaroMet cloud coverage."""

    def __init__(
        self: Self,
        hass: HomeAssistant,
        config_entry: HungaroMetConfigEntry,
        logger: logging.Logger,
    ) -> None:
        """
        Initializes a new instance of HungaroMetMsgCloudTypeDataUpdateCoordinator class.

        Args:
          hass: The Home Assistant instance.
          config_entry: The configuration entry for the integration.
          logger: The logger instance.
        """
        super().__init__(
            hass,
            logger,
            name="10 min data",
            update_interval=UPDATE_INTERVAL,
        )
        self._config_entry = config_entry
        self._api = HungaroMetMsgCloudTypeApi()
        self._latest_observation_date: datetime | None = None
        self._store = Store(
            hass, 1, f"{config_entry.domain}_{config_entry.entry_id}_cloud_data"
        )

    async def _async_update_data(self) -> list[tuple[datetime, CloudMask]]:
        """Updates stored data from data provider."""
        zone = self.hass.states.get(self._config_entry.data[CONF_ZONE])
        if not zone:
            return self.data

        latitude = zone.attributes.get(ATTR_LATITUDE)
        longitude = zone.attributes.get(ATTR_LONGITUDE)

        if latitude is None:
            return self.data

        if longitude is None:
            return self.data

        observation_dates = await self._api.get_observation_datetimes()
        exiting_dates = {d for d, _ in (self.data or [])}

        dates_to_remove = exiting_dates.difference(observation_dates)

        for date_to_remove in dates_to_remove:
            self.data.pop(date_to_remove, None)

        if self._latest_observation_date is not None:
            observation_dates = [
                d for d in observation_dates if d > self._latest_observation_date
            ]

        # For now we only look back 15 minutes (1 observation)
        # Maybe we can modify this later by storing data in dataframe a save/restore
        # as in 10 minute data
        observation_dates = observation_dates[-1:]

        new_data = list(
            zip(
                observation_dates,
                await asyncio.gather(
                    *[
                        self._api.get_cloud_mask(observation_date, latitude, longitude)
                        for observation_date in observation_dates
                    ]
                ),
                strict=True,
            )
        )

        if len(observation_dates) > 0:
            self._latest_observation_date = max(observation_dates)

        data = sorted(((self.data or []) + new_data), key=lambda item: item[0])

        try:
            await self._store.async_save(data)
        except BaseException:
            self.logger.exception("Failed to save data to store")

        return data
