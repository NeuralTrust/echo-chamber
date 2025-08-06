import os
import zoneinfo
from datetime import datetime


def get_timezone() -> zoneinfo.ZoneInfo:
    """Get the system timezone from the TZ environment variable or default to UTC.

    Returns:
        zoneinfo.ZoneInfo: The timezone object based on the TZ environment variable, or UTC if not set or invalid.
    """
    try:
        tz: zoneinfo.ZoneInfo = zoneinfo.ZoneInfo(os.getenv("TZ", "UTC"))
    except (zoneinfo.ZoneInfoNotFoundError, ValueError):
        tz = zoneinfo.ZoneInfo("UTC")
    return tz


def get_current_datetime() -> datetime:
    """Get the current datetime in the system timezone.

    Returns:
        datetime: The current datetime in the system timezone.
    """
    return datetime.now(get_timezone())
