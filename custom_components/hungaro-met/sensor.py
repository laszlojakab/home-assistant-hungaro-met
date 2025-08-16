"""Module of HungaroMet sensors."""

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, Self

from homeassistant.components.sensor import (
    RestoreSensor,
    SensorEntityDescription,
)
from homeassistant.components.sensor.const import SensorDeviceClass, SensorStateClass
from homeassistant.const import (
    CONF_ZONE,
    DEGREE,
    PERCENTAGE,
    UV_INDEX,
    EntityCategory,
    UnitOfIrradiance,
    UnitOfLength,
    UnitOfPressure,
    UnitOfSpeed,
    UnitOfTemperature,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from . import HungaroMetConfigEntry
from .const import LOGGER
from .coordinator import (
    HungaroMet10MinDataUpdateCoordinator,
    HungaroMetMsgCloudTypeDataUpdateCoordinator,
)

if TYPE_CHECKING:
    from .api import CloudMask


class HungaroMetSensor(CoordinatorEntity, RestoreSensor):
    """Represents a HungaroMet sensor which receives data from 10 minute observations."""

    def __init__(
        self: Self,
        coordinator: HungaroMet10MinDataUpdateCoordinator,
        entity_description: SensorEntityDescription,
        prop: str,
        transformer: Callable[[Any], Any] = lambda x: x,
    ) -> None:
        """
        Initializes a new instance of HungaroMetSensor class.

        Args:
          coordinator: The data update coordinator for the sensor.
          entity_description: The description of the sensor entity.
          prop: The property to observe.
          transformer: A function to transform the observed value.
        """
        super().__init__(coordinator)

        self.entity_description = entity_description
        self._observation_time = None
        self._value = None
        self._property = prop

        self._unprocessed_items = []
        self._current_item = None
        self._unsubscribe_timer = None

        self._attr_entity_registry_enabled_default = False
        self._transformer = transformer

        self._attr_entity_registry_enabled_default = False

    async def async_added_to_hass(self: Self) -> None:
        """
        Called when the entity is added to Home Assistant.

        Restores the last known state of the entity.
        """
        await super().async_added_to_hass()

        try:
            last_state = await self.async_get_last_state()
            if last_state is not None:
                observation_time = last_state.attributes.get("observation_time")

                last_sensor_data = await self.async_get_last_sensor_data()
                if last_sensor_data is not None:
                    native_value = last_sensor_data.native_value

                    self._current_item = {
                        "value": native_value,
                        "time": observation_time,
                    }
        except BaseException:  # noqa: BLE001
            LOGGER.exception("Failed to restore last state.")
            self._current_item = None

    def _process_unprocessed_data(self: Self) -> None:
        """
        Processes unprocessed data.

        Schedules the processing of next unprocessed data item (if any).
        """
        data = self.coordinator.data
        if data is None:
            return

        if self._current_item is not None and self._current_item["time"]:
            data = data.loc[data.index > self._current_item["time"]]
        else:
            data = data.iloc[[-1]]

        if len(data) == 0:
            return

        self._unprocessed_items = list(data[self._property].dropna().items())
        self._schedule_next()

    def _schedule_next(self: Self) -> None:
        """Schedules the processing of the next unprocessed data item."""
        if self._unsubscribe_timer:
            self._unsubscribe_timer()
            self._unsubscribe_timer = None

        if not self._unprocessed_items:
            return

        next_timestamp, next_value = self._unprocessed_items.pop(0)
        self._current_item = {
            "value": next_value,
            "time": next_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        self.async_write_ha_state()

        async def _delayed_update(now: datetime) -> None:  # noqa: ARG001
            self._schedule_next()

        self._unsubscribe_timer = async_call_later(self.hass, 1, _delayed_update)

    def _handle_coordinator_update(self: Self) -> None:
        self._process_unprocessed_data()

    @property
    def native_value(self: Self) -> float | None:
        """Returns the native value of the sensor."""
        return (
            self._transformer(self._current_item["value"])
            if self._current_item
            else None
        )

    @property
    def extra_state_attributes(self: Self) -> dict[str, Any]:
        """Returns the extra state attributes of the sensor."""
        if self._current_item:
            return {"observation_time": self._current_item["time"]}

        return {}


class HungaroMetCloudCoverageSensor(CoordinatorEntity, RestoreSensor):
    """Represents a HungaroMet cloud coverage sensor."""

    def __init__(
        self: Self,
        location_name: str,
        coordinator: HungaroMetMsgCloudTypeDataUpdateCoordinator,
    ) -> None:
        """
        Initializes a new instance of HungaroMetCloudCoverageSensor class.

        Args:
          location_name: The name of the location.
          coordinator: The data update coordinator.
        """
        super().__init__(coordinator)

        self._location_name = location_name
        self._observation_time = None
        self._value = None

        self.entity_description = SensorEntityDescription(
            key="cloud_coverage",
            name=f"HungaroMet - {location_name} cloud coverage",
            state_class=SensorStateClass.MEASUREMENT,
            icon="mdi:weather-cloudy",
            native_unit_of_measurement=PERCENTAGE,
        )

        self._attr_entity_registry_enabled_default = False

    async def async_added_to_hass(self: Self) -> None:
        """
        Called when the entity is added to Hass.

        Restores the last known state.
        """
        await super().async_added_to_hass()

        try:
            last_state = await self.async_get_last_state()
            if last_state is not None:
                observation_time = last_state.attributes.get("observation_time")

                last_sensor_data = await self.async_get_last_sensor_data()
                if last_sensor_data is not None:
                    native_value = last_sensor_data.native_value

                    self._observation_time = observation_time
                    self._value = native_value
        except BaseException:  # noqa: BLE001
            LOGGER.exception("Failed to restore last state")
            self._observation_time = None
            self._value = None

    @property
    def native_value(self: Self) -> float | None:
        """Returns the cloudiness in percentage."""
        data: list[tuple[datetime, CloudMask]] | None = self.coordinator.data

        if data is None:
            return None

        return data[-1][1].cloudiness

    @property
    def extra_state_attributes(self: Self) -> dict[str, Any]:
        """Returns the extra state attributes."""
        data: list[tuple[datetime, CloudMask]] | None = self.coordinator.data

        if data is None:
            return {}

        return {
            "observation_time": data[-1][0],
            "north_quadrant_cloudiness": data[-1][1].north_quadrant_cloudiness,
            "south_quadrant_cloudiness": data[-1][1].south_quadrant_cloudiness,
            "east_quadrant_cloudiness": data[-1][1].east_quadrant_cloudiness,
            "west_quadrant_cloudiness": data[-1][1].west_quadrant_cloudiness,
            "north_east_quadrant_cloudiness": data[-1][1].north_east_quadrant_cloudiness,
            "north_west_quadrant_cloudiness": data[-1][1].north_west_quadrant_cloudiness,
            "south_east_quadrant_cloudiness": data[-1][1].south_east_quadrant_cloudiness,
            "south_west_quadrant_cloudiness": data[-1][1].south_west_quadrant_cloudiness
        }


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: HungaroMetConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """
    Called when the sensor platform initialized.

    It adds HungaroMet sensors to Home Assistant.

    Args:
      hass: The Home Assistant instance.
      config_entry: The configuration entry for the HungaroMet integration.
      async_add_entities: Callback to add entities to Home Assistant.
    """
    state = hass.states.get(config_entry.data[CONF_ZONE])
    location_name = (
        state.attributes.get("friendly_name") if state else config_entry[CONF_ZONE]
    )

    coordinator_10_min = config_entry.runtime_data.coordinator_10_min
    coordinator_msg_cloud_type = config_entry.runtime_data.coordinator_msg_cloud_type

    async_add_entities(
        [
            HungaroMetSensor(
                coordinator_10_min,
                SensorEntityDescription(
                    key="temperature",
                    name=f"HungaroMet - {location_name} temperature",
                    state_class=SensorStateClass.MEASUREMENT,
                    device_class=SensorDeviceClass.TEMPERATURE,
                    entity_category=EntityCategory.DIAGNOSTIC,
                    icon="mdi:thermometer",
                    native_unit_of_measurement=UnitOfTemperature.CELSIUS,
                ),
                "ta",
            ),
            HungaroMetSensor(
                coordinator_10_min,
                SensorEntityDescription(
                    key="pressure",
                    name=f"HungaroMet - {location_name} pressure",
                    state_class=SensorStateClass.MEASUREMENT,
                    device_class=SensorDeviceClass.ATMOSPHERIC_PRESSURE,
                    entity_category=EntityCategory.DIAGNOSTIC,
                    icon="mdi:gauge",
                    native_unit_of_measurement=UnitOfPressure.HPA,
                ),
                "p",
            ),
            HungaroMetSensor(
                coordinator_10_min,
                SensorEntityDescription(
                    key="wind_speed",
                    name=f"HungaroMet - {location_name} wind speed",
                    state_class=SensorStateClass.MEASUREMENT,
                    device_class=SensorDeviceClass.WIND_SPEED,
                    entity_category=EntityCategory.DIAGNOSTIC,
                    icon="mdi:weather-windy",
                    native_unit_of_measurement=UnitOfSpeed.METERS_PER_SECOND,
                ),
                "fs",
            ),
            HungaroMetSensor(
                coordinator_10_min,
                SensorEntityDescription(
                    key="wind_speed",
                    name=f"HungaroMet - {location_name} wind gust",
                    state_class=SensorStateClass.MEASUREMENT,
                    device_class=SensorDeviceClass.WIND_SPEED,
                    entity_category=EntityCategory.DIAGNOSTIC,
                    icon="mdi:weather-windy",
                    native_unit_of_measurement=UnitOfSpeed.METERS_PER_SECOND,
                ),
                "fx",
            ),
            HungaroMetSensor(
                coordinator_10_min,
                SensorEntityDescription(
                    key="wind_direction",
                    name=f"HungaroMet - {location_name} wind direction",
                    state_class=SensorStateClass.MEASUREMENT_ANGLE,
                    device_class=SensorDeviceClass.WIND_DIRECTION,
                    entity_category=EntityCategory.DIAGNOSTIC,
                    icon="mdi:weather-windy",
                    native_unit_of_measurement=DEGREE,
                ),
                "fsd",
            ),
            HungaroMetSensor(
                coordinator_10_min,
                SensorEntityDescription(
                    key="wind_speed",
                    name=f"HungaroMet - {location_name} visibility",
                    state_class=SensorStateClass.MEASUREMENT,
                    device_class=SensorDeviceClass.DISTANCE,
                    entity_category=EntityCategory.DIAGNOSTIC,
                    icon="mdi:weather-fog",
                    native_unit_of_measurement=UnitOfLength.METERS,
                ),
                "v",
            ),
            HungaroMetSensor(
                coordinator_10_min,
                SensorEntityDescription(
                    key="rain",
                    name=f"HungaroMet - {location_name} rain",
                    state_class=SensorStateClass.MEASUREMENT,
                    device_class=SensorDeviceClass.PRECIPITATION,
                    entity_category=EntityCategory.DIAGNOSTIC,
                    icon="mdi:weather-pouring",
                    native_unit_of_measurement=UnitOfLength.MILLIMETERS,
                ),
                "r",
            ),
            HungaroMetSensor(
                coordinator_10_min,
                SensorEntityDescription(
                    key="relative_humidity",
                    name=f"HungaroMet - {location_name} relative humidity",
                    state_class=SensorStateClass.MEASUREMENT,
                    device_class=SensorDeviceClass.HUMIDITY,
                    entity_category=EntityCategory.DIAGNOSTIC,
                    icon="mdi:water-percent",
                    native_unit_of_measurement=PERCENTAGE,
                ),
                "u",
            ),
            HungaroMetSensor(
                coordinator_10_min,
                SensorEntityDescription(
                    key="uv",
                    name=f"HungaroMet - {location_name} UV index",
                    state_class=SensorStateClass.MEASUREMENT,
                    entity_category=EntityCategory.DIAGNOSTIC,
                    icon="mdi:weather-sunny-alert",
                    native_unit_of_measurement=UV_INDEX,
                ),
                "suv",
                lambda suv: round(suv * (210 / 90), 1) if suv is not None else None,
            ),
            HungaroMetSensor(
                coordinator_10_min,
                SensorEntityDescription(
                    key="global_irradiance",
                    name=f"HungaroMet - {location_name} global irradiance",
                    state_class=SensorStateClass.MEASUREMENT,
                    entity_category=EntityCategory.DIAGNOSTIC,
                    icon="mdi:weather-sunny",
                    native_unit_of_measurement=UnitOfIrradiance.WATTS_PER_SQUARE_METER,
                ),
                "sr",
            ),
            HungaroMetCloudCoverageSensor(location_name, coordinator_msg_cloud_type),
        ]
    )

    return True
