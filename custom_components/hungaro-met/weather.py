"""Module of HungaroMet weather entity."""

import math
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from homeassistant.components.weather import (
    ATTR_CONDITION_CLEAR_NIGHT,
    ATTR_CONDITION_CLOUDY,
    ATTR_CONDITION_FOG,
    ATTR_CONDITION_PARTLYCLOUDY,
    ATTR_CONDITION_POURING,
    ATTR_CONDITION_RAINY,
    ATTR_CONDITION_SNOWY,
    ATTR_CONDITION_SUNNY,
    ATTR_CONDITION_WINDY,
    CoordinatorWeatherEntity,
    WeatherEntityDescription,
)
from homeassistant.const import (
    CONF_NAME,
    UnitOfIrradiance,
    UnitOfLength,
    UnitOfPrecipitationDepth,
    UnitOfPressure,
    UnitOfSpeed,
    UnitOfTemperature,
    UnitOfVolumetricFlux,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import HungaroMetConfigEntry
from .coordinator import (
    CloudMask,
    HungaroMet10MinDataUpdateCoordinator,
    HungaroMetMsgCloudTypeDataUpdateCoordinator,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime


class HungaroMetWeatherEntity(CoordinatorWeatherEntity[HungaroMet10MinDataUpdateCoordinator]):
    """Representation of a weather entity for HungaroMet."""

    _weather_option_temperature_unit = UnitOfTemperature.CELSIUS
    _weather_option_pressure_unit = UnitOfPressure.HPA
    _weather_option_visibility_unit = UnitOfLength.METERS
    _weather_option_precipitation_unit = UnitOfPrecipitationDepth.MILLIMETERS
    _weather_option_wind_speed_unit = UnitOfSpeed.METERS_PER_SECOND

    def __init__(
        self,
        location_name: str,
        coordinator: HungaroMet10MinDataUpdateCoordinator,
        msg_cloud_type_coordinator: HungaroMetMsgCloudTypeDataUpdateCoordinator,
    ) -> None:
        """
        Initializes a new instance of HungaroMetWeatherEntity class.

        Args:
            location_name: The name of the location.
            coordinator: The data update coordinator.
            msg_cloud_type_coordinator: The cloud type data update coordinator.
        """
        CoordinatorWeatherEntity.__init__(self, coordinator)

        self._msg_cloud_type_coordinator = msg_cloud_type_coordinator

        self.entity_description = WeatherEntityDescription(
            key="weather", name=f"HungaroMet - {location_name}"
        )

        self._attr_native_precipitation_unit = UnitOfPrecipitationDepth.MILLIMETERS
        self._attr_native_pressure_unit = UnitOfPressure.HPA
        self._attr_native_temperature_unit = UnitOfTemperature.CELSIUS
        self._attr_native_visibility_unit = UnitOfLength.METERS
        self._attr_native_wind_speed_unit = UnitOfSpeed.METERS_PER_SECOND

        self._dispose_msg_cloud_type_coordinator_listener: None | Callable[[], None] = None

    @property
    def native_temperature(self: Self) -> float | None:
        """Returns the temperature in native units."""
        return self._get_prop("ta")

    @property
    def native_pressure(self: Self) -> float | None:
        """Returns the pressure in native units."""
        return self._get_prop("p")

    @property
    def native_wind_speed(self: Self) -> float | None:
        """Returns the wind speed in native units."""
        return self._get_prop("fs")

    @property
    def native_visibility(self: Self) -> float | None:
        """Returns the wind speed in native units."""
        return self._get_prop("v")

    @property
    def humidity(self: Self) -> float | None:
        """Returns the humidity in native units."""
        return self._get_prop("u")

    @property
    def wind_bearing(self: Self) -> float | str | None:
        """Returns the wind bearing."""
        return self._get_prop("fsd")

    @property
    def native_wind_gust_speed(self: Self) -> float | None:
        """Returns the wind gust speed in native units."""
        return self._get_prop("fx")

    @property
    def uv_index(self: Self) -> float | None:
        """Returns the UV index."""
        suv = self._get_prop("suv")

        if suv is None:
            return None

        d_med = 210
        return round(suv * (d_med / 90), 1)

    @property
    def native_dew_point(self: Self) -> float | None:
        """Returns the dew point temperature in native units."""
        if self.native_temperature is None or self.humidity is None:
            return None

        a = 17.27
        b = 237.7
        alpha = ((a * self.native_temperature) / (b + self.native_temperature)) + math.log(
            self.humidity / 100.0
        )

        return (b * alpha) / (a - alpha)

    @property
    def native_apparent_temperature(self: Self) -> float | None:
        """Returns the apparent temperature in native units."""
        wind_speed = self.native_wind_speed
        temperature = self.native_temperature
        relative_humidity = self.humidity

        if wind_speed is None or temperature is None or relative_humidity is None:
            return None

        wind_speed_kmh = wind_speed * 3.6

        if temperature >= 27:  # noqa: PLR2004
            # Heat Index (valid for hot/humid conditions)
            return (
                -8.784695
                + 1.61139411 * temperature
                + 2.338549 * relative_humidity
                - 0.14611605 * temperature * relative_humidity
                - 0.01230809 * temperature**2
                - 0.01642482 * relative_humidity**2
                + 0.00221173 * temperature**2 * relative_humidity
                + 0.00072546 * temperature * relative_humidity**2
                - 0.00000358 * temperature**2 * relative_humidity**2
            )

        if temperature <= 10 and wind_speed_kmh >= 4.8:  # noqa: PLR2004
            # Wind chill (valid for cold/windy conditions)
            return (
                13.12
                + 0.6215 * temperature
                - 11.37 * wind_speed_kmh**0.16
                + 0.3965 * temperature * wind_speed_kmh**0.16
            )

        return temperature  # No correction needed

    @property
    def precipitation_intensity(self: Self) -> float | None:
        """Returns the precipitation intensity in mm/h."""
        r = self._get_prop("r")

        if r is None:
            return None

        return r * 6

    @property
    def precipitation_intensity_unit_of_measurement(self: Self) -> str:
        """Returns the unit of measurement for precipitation intensity."""
        return UnitOfVolumetricFlux.MILLIMETERS_PER_HOUR

    @property
    def global_irradiance(self: Self) -> float | None:
        """Returns the global irradiance in W/m²."""
        return self._get_prop("sr")

    @property
    def cloud_coverage(self: Self) -> int | None:
        """Returns the cloud coverage in percentage."""
        data: list[tuple[datetime, CloudMask]] | None = self._msg_cloud_type_coordinator.data

        if data is None:
            return None

        return int(data[-1][1].cloudiness)

    @property
    def irradiance_unit_of_measurement(self: Self) -> str:
        """Returns the unit of measurement for irradiance."""
        return UnitOfIrradiance.WATTS_PER_SQUARE_METER

    @property
    def extra_state_attributes(self: Self) -> dict[str, Any]:
        """Returns extra state attributes."""
        return {
            "precipitation_intensity": self.precipitation_intensity,
            "precipitation_intensity_unit_of_measurement": self.precipitation_intensity_unit_of_measurement,  # noqa: E501
            "global_irradiance": self.global_irradiance,
            "irradiance_unit_of_measurement": self.irradiance_unit_of_measurement,
        }

    @property
    def condition(self: Self) -> str | None:
        """Returns the current condition."""
        condition: str | None = None
        if (
            self.native_visibility is not None and self.native_visibility < 1000  # noqa: PLR2004
        ):
            condition = ATTR_CONDITION_FOG
        elif (
            self.precipitation_intensity is not None and self.precipitation_intensity > 2  # noqa: PLR2004
        ):
            condition = ATTR_CONDITION_POURING
        elif self.precipitation_intensity is not None and self.precipitation_intensity > 0:
            if self.native_temperature is not None and self.native_temperature <= 1:
                condition = ATTR_CONDITION_SNOWY
            else:
                condition = ATTR_CONDITION_RAINY
        elif (
            self.native_wind_speed is not None and self.native_wind_speed > 15  # noqa: PLR2004
        ):
            condition = ATTR_CONDITION_WINDY
        else:
            sun_state = self.coordinator.hass.states.get("sun.sun")

            if self.cloud_coverage is None:
                return None

            if self.cloud_coverage > 70:  # noqa: PLR2004
                condition = ATTR_CONDITION_CLOUDY
            elif self.cloud_coverage > 30:  # noqa: PLR2004
                condition = ATTR_CONDITION_PARTLYCLOUDY
            else:
                is_day = sun_state is None or sun_state.state == "above_horizon"

                condition = ATTR_CONDITION_SUNNY if is_day else ATTR_CONDITION_CLEAR_NIGHT

        return condition

    def _get_prop(self, prop: str) -> float | None:
        if self.coordinator.data is None:
            return None

        value = self.coordinator.data[prop].iloc[-1]

        return None if np.isnan(value) else value

    async def async_added_to_hass(self: Self) -> None:
        """Call when the entity is added to hass."""
        await super().async_added_to_hass()

        self._dispose_msg_cloud_type_coordinator_listener = (
            self._msg_cloud_type_coordinator.async_add_listener(
                self._handle_msg_cloud_type_coordinator_update
            )
        )

    async def async_will_remove_from_hass(self: Self) -> None:
        """Call when the entity is removed from hass."""
        self._dispose_msg_cloud_type_coordinator_listener()

        await super().async_will_remove_from_hass()

    def _handle_msg_cloud_type_coordinator_update(self: Self) -> None:
        """Handle updates from the cloud type coordinator."""
        self.async_write_ha_state()


async def async_setup_entry(
    hass: HomeAssistant,  # noqa: ARG001
    config_entry: HungaroMetConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> bool:
    """Call when the entity is added to Home assistant."""
    async_add_entities(
        [
            HungaroMetWeatherEntity(
                config_entry.data.get(CONF_NAME),
                config_entry.runtime_data.coordinator_10_min,
                config_entry.runtime_data.coordinator_msg_cloud_type,
            )
        ]
    )

    return True
