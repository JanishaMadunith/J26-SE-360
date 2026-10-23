"""J26-SE-360 shared contracts. Change a contract here, run `make contracts`, and every consumer
(core API, module services, Flutter app via JSON Schema) stays in sync."""

from .common import AffectiveLabel, Contract, ModelInfo, TimeWindow, utc_now  # noqa: F401
from .m1_bridge import *  # noqa: F401,F403
from .modules import *  # noqa: F401,F403
from .sensing import *  # noqa: F401,F403
from .tasks import *  # noqa: F401,F403

__version__ = "0.1.0"

