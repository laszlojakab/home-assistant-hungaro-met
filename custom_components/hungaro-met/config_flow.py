"""The configuration flow for HungaroMet integration."""

from typing import Any, Self

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import CONF_NAME, CONF_RADIUS, CONF_ZONE
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import CLIMATE_TYPE, CONF_CLIMATE, DOMAIN


def _get_schema(
    zone: str | None = None,
    climate: CLIMATE_TYPE | None = None,
    radius: int = 50,
    **kwargs: dict[str, Any],  # noqa: ARG001
) -> vol.Schema:
    return vol.Schema(
        {
            vol.Required(CONF_ZONE, default=zone): selector.EntitySelector(
                selector.EntitySelectorConfig(domain=["zone"]),
            ),
            vol.Required(CONF_CLIMATE, default=climate): vol.In(
                {
                    "humid": "Humid",
                    "sub-humid": "Sub-humid",
                    "semi-arid": "Semi-arid",
                    "arid": "Arid",
                }
            ),
            vol.Required(CONF_RADIUS, default=radius): int,
        }
    )


class HungaroMetOptionsFlow(config_entries.OptionsFlowWithConfigEntry):
    """The options flow handler for the Door and window integration."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """
        Initializes a new instance of HungaroMetOptionsFlow class.

        Args:
            config_entry: The config entry to use for the options flow.
        """
        super().__init__(config_entry)
        self.updated_config = {}
        self.data: dict[str, any] = {}

    async def async_step_init(
        self: Self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """
        The initialization step of the options flow.

        Args:
            user_input: The user input from the options flow.

        Returns:
            The result of the options flow.
        """
        if user_input is not None:
            self.hass.config_entries.async_update_entry(
                self.config_entry, data=self.config_entry.data | user_input
            )

            return self.async_abort(reason="reconfigure_successful")

        return self.async_show_form(
            step_id="init", data_schema=_get_schema(**self.config_entry.data)
        )


@config_entries.HANDLERS.register(DOMAIN)
class HungaroMetConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Configuration flow handler for HungaroMet integration."""

    VERSION = 1

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigFlow,
    ) -> config_entries.OptionsFlowWithConfigEntry:
        """
        Gets the options flow.

        Args:
            config_entry: The config entry to use for the options flow.

        Returns:
            The options flow.
        """
        return HungaroMetOptionsFlow(config_entry)

    async def async_step_user(
        self: Self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """
        Handles the step when integration added from the UI.

        Args:
            user_input: The user input from the config flow.

        Returns:
            The result of the config flow.
        """
        if user_input is not None:
            await self.async_set_unique_id(user_input[CONF_ZONE])

            self._abort_if_unique_id_configured()

            state = self.hass.states.get(user_input[CONF_ZONE])
            if state:
                friendly_name = state.attributes.get("friendly_name")
            else:
                friendly_name = user_input[CONF_ZONE]

            return self.async_create_entry(
                title=f"HungaroMet [{friendly_name}]",
                data={**user_input, CONF_NAME: friendly_name},
            )

        return self.async_show_form(
            step_id="user",
            data_schema=_get_schema(),
        )
