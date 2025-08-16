"""Module for server access."""

import asyncio
import io
import zipfile
from collections.abc import Iterable
from datetime import UTC, datetime
from http import HTTPStatus
from io import BytesIO
from logging import Logger, getLogger
from typing import Final, Literal, Self

import aiohttp
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from cachetools import TTLCache
from PIL import Image
from pyproj import Geod
from scipy.spatial import Delaunay

type Quadrant = Literal["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
"""Represents the eight quadrants of a compass."""


class _GeoPixelCoordinateTransformer:
    """Transforms geographical coordinates to pixel coordinates and vice versa."""

    _ref_points: Final[list[tuple[float, float, int, int]]] = [
        (50, 20, 1052, 481),
        (40, 20, 1074, 828),
        (50, 10, 830, 481),
        (40, 10, 807, 828),
    ]
    """
    The reference points for calibration of the MSG Cloud Type image.
    The the first two values are the latitude and longitude of the reference point,
    and the last two values are the pixel coordinates in the image.
    """

    def __init__(self: Self) -> None:
        """Initializes the GeoPixelCoordinateTransformer instance."""
        self._ax, self._ay = self._compute_affine_transform(self._ref_points)
        """
        Affine transform parameters for converting pixel coordinates to latitude and longitude.
        """
        self._lat_params, self._lon_params = self._compute_inverse_affine(
            self._ref_points
        )

    def to_pixel_coordinate(
        self: Self, latitude: float, longitude: float
    ) -> tuple[int, int]:
        """Transforms geographical coordinates to pixel coordinates."""
        x = self._ax[0] * latitude + self._ax[1] * longitude + self._ax[2]
        y = self._ay[0] * latitude + self._ay[1] * longitude + self._ay[2]

        return x, y

    def to_geo_coordinate(self: Self, x: int, y: int) -> tuple[float, float]:
        """Transforms pixel coordinates to geographical coordinates."""
        latitude = (
            self._lat_params[0] * x + self._lat_params[1] * y + self._lat_params[2]
        )
        longitude = (
            self._lon_params[0] * x + self._lon_params[1] * y + self._lon_params[2]
        )

        return latitude, longitude

    def _compute_affine_transform(
        self: Self, ref_points: list[tuple[float, float, int, int]]
    ) -> tuple[np.ndarray, np.ndarray]:
        coefficients = []
        x_coordinates = []
        y_coordinates = []
        for lat, lon, px, py in ref_points:
            coefficients.append([lat, lon, 1])
            x_coordinates.append(px)
            y_coordinates.append(py)

        ax = np.linalg.lstsq(coefficients, x_coordinates, rcond=None)[0]
        ay = np.linalg.lstsq(coefficients, y_coordinates, rcond=None)[0]
        return ax, ay

    def _compute_inverse_affine(
        self: Self, ref_points: list[tuple[float, float, int, int]]
    ) -> tuple[np.ndarray, np.ndarray]:
        coefficients = []
        latitudes = []
        longitudes = []
        for lat, lon, px, py in ref_points:
            coefficients.append([px, py, 1])
            latitudes.append(lat)
            longitudes.append(lon)

        lat_params = np.linalg.lstsq(coefficients, latitudes, rcond=None)[0]
        lon_params = np.linalg.lstsq(coefficients, longitudes, rcond=None)[0]
        return lat_params, lon_params


class CloudMask:
    """Represents a cloud mask."""

    _quadrant_definitions: Final[dict[Quadrant, list[tuple[float, float]]]] = {
        "N": [(67.5, 112.5)],
        "NE": [(22.5, 67.5)],
        "E": [(337.5, 360), (0, 22.5)],
        "SE": [(292.5, 337.5)],
        "S": [(247.5, 292.5)],
        "SW": [(202.5, 247.5)],
        "W": [(157.5, 202.5)],
        "NW": [(112.5, 157.5)],
    }
    """
    The quadrants definitions. The key is the name of the quadrant
    and the value is a list of angle ranges (start, end) for that quadrant.
    """

    def __init__(self: Self, mask: np.ndarray) -> None:
        """
        Initializes the CloudMask instance.

        Args:
          mask: A numpy array representing the cloud mask.
        """
        self.mask = mask
        self._cloud_coverage = None
        self._north_quadrant_cloud_coverage = None
        self._north_east_quadrant_cloud_coverage = None
        self._east_quadrant_cloud_coverage = None
        self._south_east_quadrant_cloud_coverage = None
        self._south_quadrant_cloud_coverage = None
        self._south_west_quadrant_cloud_coverage = None
        self._west_quadrant_cloud_coverage = None
        self._north_west_quadrant_cloud_coverage = None

    @property
    def width(self: Self) -> int:
        """
        Returns the width of the cloud mask.

        Returns:
          The width of the cloud mask.
        """
        return self.mask.shape[1]

    @property
    def height(self: Self) -> int:
        """
        Returns the height of the cloud mask.

        Returns:
          The height of the cloud mask.
        """
        return self.mask.shape[0]

    @property
    def cloudiness(self: Self) -> float | None:
        """Returns the cloudiness percentage."""
        if self._cloud_coverage is None:
            known_values = self.mask[self.mask >= 0]
            if known_values.size == 0:
                return None

            self._cloud_coverage = np.mean(known_values).round()

        return self._cloud_coverage

    @property
    def north_quadrant_cloud_coverage(self: Self) -> float | None:
        """Returns the cloudiness percentage of the north quadrant."""
        if self._north_quadrant_cloud_coverage is None:
            self._north_quadrant_cloud_coverage = self._quadrant_cloud_cover("N")
        return self._north_quadrant_cloud_coverage

    @property
    def north_east_quadrant_cloud_coverage(self: Self) -> float | None:
        """Returns the cloudiness percentage of the north-east quadrant."""
        if self._north_east_quadrant_cloud_coverage is None:
            self._north_east_quadrant_cloud_coverage = self._quadrant_cloud_cover("NE")
        return self._north_east_quadrant_cloud_coverage

    @property
    def north_west_quadrant_cloud_coverage(self: Self) -> float | None:
        """Returns the cloudiness percentage of the north-west quadrant."""
        if self._north_west_quadrant_cloud_coverage is None:
            self._north_west_quadrant_cloud_coverage = self._quadrant_cloud_cover("NW")
        return self._north_west_quadrant_cloud_coverage

    @property
    def east_quadrant_cloud_coverage(self: Self) -> float | None:
        """Returns the cloud coverage percentage of the east quadrant."""
        if self._east_quadrant_cloud_coverage is None:
            self._east_quadrant_cloud_coverage = self._quadrant_cloud_cover("E")
        return self._east_quadrant_cloud_coverage

    @property
    def south_east_quadrant_cloud_coverage(self: Self) -> float | None:
        """Returns the cloud coverage percentage of the south-east quadrant."""
        if self._south_east_quadrant_cloud_coverage is None:
            self._south_east_quadrant_cloud_coverage = self._quadrant_cloud_cover("SE")
        return self._south_east_quadrant_cloud_coverage

    @property
    def south_quadrant_cloud_coverage(self: Self) -> float | None:
        """Returns the cloud coverage percentage of the south quadrant."""
        if self._south_quadrant_cloud_coverage is None:
            self._south_quadrant_cloud_coverage = self._quadrant_cloud_cover("S")
        return self._south_quadrant_cloud_coverage

    @property
    def south_west_quadrant_cloud_coverage(self: Self) -> float | None:
        """Returns the cloud coverage percentage of the south-west quadrant."""
        if self._south_west_quadrant_cloud_coverage is None:
            self._south_west_quadrant_cloud_coverage = self._quadrant_cloud_cover("SW")
        return self._south_west_quadrant_cloud_coverage

    @property
    def west_quadrant_cloud_coverage(self: Self) -> float | None:
        """Returns the cloud coverage percentage of the west quadrant."""
        if self._west_quadrant_cloud_coverage is None:
            self._west_quadrant_cloud_coverage = self._quadrant_cloud_cover("W")
        return self._west_quadrant_cloud_coverage

    @property
    def data(self: Self) -> np.ndarray:
        """
        Returns the underlying numpy array of the cloud mask.

        Returns:
          The numpy array representing the cloud mask.
        """
        return self.mask

    def _quadrant_cloud_cover(self: Self, quadrant: Quadrant) -> float | None:
        h, w = self.data.shape
        cx, cy = w // 2, h // 2

        y_indices, x_indices = np.ogrid[:h, :w]
        dx = x_indices - cx
        dy = cy - y_indices

        angles = (np.degrees(np.arctan2(dy, dx)) + 360) % 360

        sector_def = self._quadrant_definitions[quadrant]

        sector_mask = np.zeros_like(self.data, dtype=bool)
        for angle_min, angle_max in sector_def:
            angle_mask = (angles >= angle_min) & (angles < angle_max)
            sector_mask |= angle_mask

        values = self.data[sector_mask]
        values = values[(values >= 0)]

        return np.round(values.mean()) if values.size > 0 else 0


class HungaroMetMsgCloudTypeApi:
    """The API for the Hungarian Meteorological Service's MSG Cloud Type data."""

    _unknown: Final[int] = -1
    _border: Final[int] = -2

    _base_url: Final[str] = "https://odp.met.hu/weather/satellite/MSG/png/CloudType"
    """The base URL for the MSG Cloud Type data."""

    _ref_points: Final[list[tuple[float, float, int, int]]] = [
        (50, 20, 1052, 481),
        (40, 20, 1074, 828),
        (50, 10, 830, 481),
        (40, 10, 807, 828),
    ]
    """
    The reference points for calibration of the MSG Cloud Type image.
    The the first two values are the latitude and longitude of the reference point,
    and the last two values are the pixel coordinates in the image.
    """

    _cloud_percent_map: Final[dict[tuple[int, int, int], int]] = {
        (198, 0, 198): 30,  # Vízfelhő és derült felszín keveréke
        (
            170,
            88,
            173,
        ): 30,  # Vízfelhő és derült felszín keveréke szélesség, hosszúság vonal
        (89, 202, 157): 50,  # Cirrus alacsony vagy középszintű felhő felett
        (
            125,
            171,
            156,
        ): 50,  # Cirrus alacsony vagy középszintű felhő felett szélesség, hosszúság vonal
        (0, 242, 238): 30,  # Vastag cirrus
        (88, 188, 189): 30,  # Vastag cirrus szélesség, hosszúság vonal
        (0, 182, 230): 40,  # Közepesen vastag cirrus
        (88, 163, 186): 40,  # Közepesen vastag cirrus szélesség, hosszúság vonal
        (0, 80, 214): 20,  # Vékony cirrus
        (88, 121, 179): 20,  # Vékony cirrus szélesség, hosszúság vonal
        (230, 230, 230): 100,  # Nagyon magas, vastag felhő
        (214, 214, 149): 100,  # Magas, vastag felhő
        (176, 176, 152): 100,  # Magas, vastag felhő szélesség, hosszúság vonal
        (238, 242, 0): 75,  # Középszintű felhő
        (186, 188, 91): 75,  # Középszintű felhő szélesség, hosszúság vonal
        (255, 182, 0): 80,  # Alacsony felhő
        (193, 163, 91): 80,  # Alacsony felhő szélesség, hosszúság vonal
        (255, 101, 0): 55,  # Nagyon alacsony felhő
        (193, 130, 91): 55,  # Nagyon alacsony felhő szélesség, hosszúság vonal
        (222, 161, 222): 0,  # Hóval, jéggel borított tenger
        (246, 190, 246): 0,  # Hóval borított szárazföld
        (0, 0, 0): 0,  # derült tenger
        (88, 88, 91): 0,  # derült tenger szélesség, hosszúság vonal
        (88, 138, 91): 0,  # derült szárazföld szélesség, hosszúság vonal
        (124, 144, 127): 0,  # derült szárazföld szélesség, hosszúság vonal duplán
        (0, 121, 0): 0,  # derült szárazföld
        (250, 250, 231): -2,  # országhatár
        (191, 191, 186): -2,  # országhatár szélesség, hosszúság vonal
    }
    """
    The mapping of RGB values to cloud percentage.

    Value -1 is used to represent uknown cloud percentage and
    -2 is used to represent the country border.
    """

    def __init__(self: Self) -> None:
        """Initializes the HungaroMetMsgCloudTypeApi instance."""
        self._coordinate_transformer = _GeoPixelCoordinateTransformer()
        """The transformer to transform between pixel and geographic coordinates."""
        self._geod = Geod(ellps="WGS84")
        """Geod object for calculating distances and angles on the WGS84 ellipsoid."""

    async def download_image(
        self: Self, observation_time: datetime
    ) -> Image.Image | None:
        """
        Downloads the MSG Cloud Type image for the given observation time.

        Args:
          observation_time: The time of the observation.

        Returns:
          An Image object if the image exists, or None if the image does not exist.
        """
        file_name = (
            f"satellite_MSG-CloudType-{observation_time.strftime('%Y%m%d_%H%M')}.png"
        )
        return await self._download_image(file_name)

    async def get_observation_datetimes(self: Self) -> list[datetime]:
        """Gets the observation datetimes."""
        async with (
            aiohttp.ClientSession() as session,
            session.get(self._base_url + "?F=0;C=N;O=D") as resp,
        ):
            resp.raise_for_status()
            return sorted(
                [
                    datetime.strptime(a.get("href")[24:37], "%Y%m%d_%H%M").replace(
                        tzinfo=UTC
                    )
                    for a in BeautifulSoup(await resp.text(), "html.parser").find_all(
                        "a"
                    )
                    if a.get("href") is not None and a.get("href").endswith(".png")
                ]
            )

    async def get_cloud_mask(
        self: Self,
        observation_time: datetime,
        latitude: float,
        longitude: float,
        radius: float = 300,
    ) -> CloudMask | None:
        """
        Gets the cloud mask for the given observation time.

        Args:
          observation_time: The time of the observation.
          latitude: The latitude of the point to get the cloud mask for.
          longitude: The longitude of the point to get the cloud mask for.
          radius: The radius around the point to get the cloud mask for (in kilometers).

        Returns:
          A CloudMask object if the cloud mask could be obtained, or None if it could not.
        """
        image = await self.download_image(observation_time)
        if image is None:
            return None

        # Affine transformation parameters

        img_arr = np.array(image)
        height, width, _ = img_arr.shape

        cx, cy = self._coordinate_transformer.to_pixel_coordinate(latitude, longitude)

        mask = np.full((height, width), self._unknown, dtype=float)

        # Pixel bejárás csak a kör sugarában
        x_min = max(int(cx - radius), 0)
        x_max = min(int(cx + radius), width - 1)
        y_min = max(int(cy - radius), 0)
        y_max = min(int(cy + radius), height - 1)

        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                # Skip legend pixels
                if x >= 1246 and y >= 481:  # noqa: PLR2004
                    continue

                lat_p, lon_p = self._coordinate_transformer.to_geo_coordinate(x, y)
                _, _, dist_m = self._geod.inv(longitude, latitude, lon_p, lat_p)

                if dist_m <= radius * 1000:
                    rgb = tuple(img_arr[y, x])
                    mask[y, x] = self._cloud_percent_map.get(rgb, self._unknown)

        return CloudMask(self._crop(mask))

    def convert_cloud_mask_to_image(self: Self, cloud_mask: CloudMask) -> Image.Image:
        """
        Converts the cloud mask to an image.

        Args:
          cloud_mask: The CloudMask object to convert.

        Returns:
          An Image object representing the cloud mask.
        """
        img_data = np.zeros((cloud_mask.height, cloud_mask.width, 3), dtype=np.uint8)

        # 0..100 -> gray scale
        mask_normal = (cloud_mask.data >= 0) & (cloud_mask.data <= 100)  # noqa: PLR2004
        gray_vals = (cloud_mask.data[mask_normal] * 255 / 100).astype(np.uint8)
        img_data[mask_normal] = np.stack([gray_vals] * 3, axis=1)

        mask_unknown = cloud_mask.data == self._unknown
        img_data[mask_unknown] = [112, 150, 149]

        mask_border = cloud_mask.data == self._border
        img_data[mask_border] = [165, 203, 204]

        img_data = img_data.astype(np.uint8)

        return Image.fromarray(img_data.astype(np.uint8))

    async def _download_image(self: Self, file: str) -> Image.Image | None:
        """
        Downloads the specified image file from the server.

        Args:
          file: The name of the image file to download.

        Returns:
          An Image object if the file exists, or None if the file does not exist.
        """
        async with (
            aiohttp.ClientSession() as session,
            session.get(f"{self._base_url}/{file}") as resp,
        ):
            if resp.status == HTTPStatus.NOT_FOUND:
                return None

            resp.raise_for_status()
            data = await resp.read()

            return Image.open(BytesIO(data)).convert("RGB")

    def _crop(self: Self, mask: np.ndarray) -> np.ndarray:
        """Crops the mask to the bounding box of non unknown values."""
        non_101_indices = np.where(mask != -1)

        # Get the minimum and maximum row and column indices
        min_row, max_row = np.min(non_101_indices[0]), np.max(non_101_indices[0])
        min_col, max_col = np.min(non_101_indices[1]), np.max(non_101_indices[1])

        # Extract the sub-matrix
        return mask[min_row : max_row + 1, min_col : max_col + 1]


class _HungaroMet10MinRawDataProvider:
    _meta_url: Final[str] = (
        "https://odp.met.hu/climate/observations_hungary/10_minutes/station_meta_auto.csv"
    )
    _base_url: Final[str] = (
        "https://odp.met.hu/climate/observations_hungary/10_minutes/now"
    )

    def __init__(self, logger: Logger | None = None) -> None:
        self._logger = logger or getLogger(__name__)

    async def get_stations(self: Self) -> pd.DataFrame | None:
        async with (
            aiohttp.ClientSession() as session,
            session.get(self._meta_url) as response,
        ):
            self._logger.debug("Getting stations from %s.", self._meta_url)

            if response.status != HTTPStatus.OK:
                self._logger.error(
                    "Failed to get stations. Status: %s", response.status
                )
                return None

            text = await response.text()

            loop = asyncio.get_running_loop()

            def parse_csv() -> pd.DataFrame:
                csv_content = (
                    pd.read_csv(
                        io.StringIO(text),
                        sep=";",
                        comment="#",
                    )
                    .rename(columns=lambda x: x.strip())
                    .set_index("StationNumber")
                    .drop(columns=["EOR"])
                )

                result = csv_content.loc[
                    csv_content["EndDate"]
                    >= int(pd.Timestamp.today().normalize().strftime("%Y%m%d"))
                ]

                self._logger.debug("Total %d stations loaded.", len(result))

                return result

            return await loop.run_in_executor(None, parse_csv)

    async def get_station_data(
        self: Self,
        station_number: int,
        previous_version: tuple[str, pd.DataFrame] | None = None,
    ) -> tuple[str, pd.DataFrame] | None:
        url = f"{self._base_url}/HABP_10M_{station_number}_now.zip"

        self._logger.debug("Getting station data for %s.", station_number)

        response_content: bytes
        response_status: int
        if previous_version is not None:
            (etag, _) = previous_version
            async with (
                aiohttp.ClientSession(headers={"If-None-Match": etag}) as session,
                session.get(url) as response,
            ):
                if response.status == HTTPStatus.NOT_MODIFIED:
                    self._logger.debug(
                        "Station data not modified, returning stored version for %s.",
                        station_number,
                    )
                    return previous_version

                response_status = response.status
                response_content = await response.read()
        else:
            async with aiohttp.ClientSession() as session, session.get(url) as response:
                response_status = response.status
                response_content = await response.read()

        if response_status != HTTPStatus.OK:
            self._logger.error(
                "Failed to get station data for %s. Status: %s",
                station_number,
                response_status,
            )
            return None

        with zipfile.ZipFile(io.BytesIO(response_content)) as z:
            # Assuming there's only one file or you know the name of the csv file
            csv_filename = z.namelist()[0]
            if datetime.now(UTC) - datetime(
                *z.getinfo(csv_filename).date_time, tzinfo=UTC
            ) > pd.Timedelta(minutes=60):
                return None

            with z.open(csv_filename) as f:
                csv_content = (
                    pd.read_csv(
                        io.StringIO(f.read().decode("utf-8")), sep=";", comment="#"
                    )
                    .rename(columns=lambda x: x.strip())
                    .assign(
                        Time=lambda df: pd.to_datetime(
                            df["Time"], format="%Y%m%d%H%M", utc=True
                        )
                    )
                    .set_index(["StationNumber", "Time"])
                    .drop(columns=["EOR"])
                )

                csv_content = csv_content.drop(
                    columns=[col for col in csv_content.columns if col.startswith("Q_")]
                ).replace(-999, np.nan)

                return (response.headers.get("ETag"), csv_content)


class _HungaroMetDataProvider:
    def __init__(
        self: Self,
        raw_data_provider: _HungaroMet10MinRawDataProvider,
        logger: Logger | None = None,
    ) -> None:
        self._logger = logger or getLogger(__name__)
        self._station_data_cache = TTLCache(maxsize=500, ttl=600)
        self._short_term_station_number_cache = TTLCache(maxsize=500, ttl=120)
        self._raw_data_provider = raw_data_provider

    async def get_stations(self: Self) -> pd.DataFrame | None:
        return await self._raw_data_provider.get_stations()

    async def get_station_data(self: Self, station_number: int) -> pd.DataFrame | None:
        data_from_cache = self._station_data_cache.get(station_number, None)

        if (
            data_from_cache is not None
            and self._short_term_station_number_cache.get(station_number, None)
            is not None
        ):
            _, data = data_from_cache
            return data

        result = await self._raw_data_provider.get_station_data(
            station_number, self._station_data_cache.get(station_number, None)
        )

        if result is None:
            return None

        self._station_data_cache[station_number] = result
        self._short_term_station_number_cache[station_number] = True

        return result[1]

    async def get_location_data(
        self: Self,
        latitude: float,
        longitude: float,
        properties: Iterable[str],
        max_radius: float,
        after_timestamp: pd.Timestamp | None = None,
    ) -> pd.DataFrame | None:
        self._logger.debug(
            "Getting location data. "
            "latitude=%s, longitude=%s, properties=%s, max_radius=%s, after_timestamp=%s",
            latitude,
            longitude,
            properties,
            max_radius,
            after_timestamp,
        )
        stations = await self.get_stations()

        if stations is None:
            self._logger.error(
                "Failed to get location data, could not load stations. "
                "latitude=%s, longitude=%s, properties=%s, max_radius=%s, after_timestamp=%s",
                latitude,
                longitude,
                properties,
                max_radius,
                after_timestamp,
            )
            return None

        # Filter stations based on distance
        stations["distance"] = self._haversine(
            stations["Latitude"], stations["Longitude"], latitude, longitude
        )

        stations = stations.loc[stations["distance"] <= max_radius]

        self._logger.debug(
            "%d kept after filtering based on distance (%s km).",
            len(stations),
            max_radius,
        )

        time_index = pd.date_range(
            start=(
                after_timestamp
                if after_timestamp is not None
                else (pd.Timestamp.now(UTC).normalize() - pd.Timedelta(days=1))
            ),
            end=pd.Timestamp.now(UTC).floor(freq="10min") - pd.Timedelta(minutes=10),
            freq="10min",
        )

        target = np.array([latitude, longitude])
        interpolated = {}

        for prop in properties:
            stations_for_property = stations
            interpolated[prop] = []
            for time in time_index:
                self._logger.debug(
                    "Interpolating data for `%s` property at %s.", prop, time
                )
                while True:
                    station_coords = stations_for_property[
                        ["Latitude", "Longitude"]
                    ].to_numpy()

                    tri = Delaunay(station_coords)
                    simplex = tri.find_simplex(target)

                    if simplex >= 0:
                        vertices = tri.simplices[simplex]
                        station_numbers = list(
                            stations_for_property.iloc[vertices].index
                        )

                        async def get_station_data(
                            station_number: int, time: pd.Timestamp, prop: str
                        ) -> float:
                            data = await self.get_station_data(station_number)
                            if data is None:
                                return np.nan

                            row = data.loc[data.index.get_level_values("Time") == time][
                                prop
                            ]
                            if len(row) > 0:
                                return row.iloc[0]

                            return np.nan

                        tasks = [
                            get_station_data(station_number, time, prop)
                            for station_number in station_numbers
                        ]
                        tri_values = list(await asyncio.gather(*tasks))

                        nans = np.isnan(tri_values)
                        if nans.sum() > 0:
                            nan_station_numbers = np.array(station_numbers)[nans]
                            stations_for_property = stations_for_property.drop(
                                nan_station_numbers
                            )
                            continue

                        tri_points = station_coords[vertices]

                        interpolated_value = round(
                            self._barycentric_interpolation(
                                target, tri_points, tri_values
                            ),
                            2,
                        )

                        interpolated[prop].append(interpolated_value)

                        self._logger.debug(
                            "Data Interpolating data for `%s` property "
                            "at %s completed. Used stations: %s.",
                            prop,
                            time,
                            [
                                str(stations.loc[station_number]["StationName"]).strip()
                                + " ("
                                + str(
                                    round(stations.loc[station_number]["distance"], 1)
                                )
                                + " km)"
                                for station_number in station_numbers
                            ],
                        )
                        break

                    self._logger.warning(
                        "Could not interpolate data for `%s` property. No simplex solution.",
                        prop,
                    )
                    interpolated[prop] = np.nan
                    break

        return pd.DataFrame(interpolated, index=time_index)

    def _haversine(
        self: Self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """
        Calculates the distance in km between to coordinates.

        Args:
          lat1: Latitude of the first point.
          lon1: Longitude of the first point.
          lat2: Latitude of the second point.
          lon2: Longitude of the second point.

        Returns:
          Distance in km between the two points.
        """
        earth_radius_in_meters = 6371.0
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)

        a = (
            np.sin(dphi / 2.0) ** 2
            + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
        )
        return 2 * earth_radius_in_meters * np.arcsin(np.sqrt(a))

    def _barycentric_interpolation(
        self: Self, target_point: np.ndarray, tri_points: np.ndarray, values: np.ndarray
    ) -> float:
        """
        Performs barycentric interpolation.

        Args:
          target_point: The point to interpolate.
          tri_points: The points of the triangle.
          values: The values at the triangle points.

        Returns:
          The interpolated value.
        """
        coefficients = np.array(
            [
                [tri_points[0, 0], tri_points[0, 1], 1],
                [tri_points[1, 0], tri_points[1, 1], 1],
                [tri_points[2, 0], tri_points[2, 1], 1],
            ]
        )
        dependent = np.array([target_point[0], target_point[1], 1])
        w = np.linalg.solve(coefficients.T, dependent)
        return np.dot(w, values)


class HungaroMet10MinDataProvider(_HungaroMetDataProvider):
    """HungaroMet 10 minute data provider."""

    def __init__(self: Self, logger: Logger | None = None) -> None:
        """Initializes a new instance of HungaroMet10MinDataProvider class."""
        super().__init__(_HungaroMet10MinRawDataProvider(logger), logger)
