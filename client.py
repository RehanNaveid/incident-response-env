"""
Client wrapper for the Incident Response Environment.

Provides ``IncidentResponseEnv``, a typed ``EnvClient`` subclass that can
connect to a running environment server (local Docker or remote Space).
"""

from openenv.core.env_client import EnvClient

from models import IncidentAction, IncidentObservation


class IncidentResponseClient(EnvClient[IncidentAction, IncidentObservation]):
    """Typed async client for the Incident Response Environment.

    Example (async)::

        async with IncidentResponseClient(base_url="http://localhost:7860") as client:
            result = await client.reset()
            print(result.observation)
            result = await client.step(IncidentAction(action="block ip 203.0.113.45"))

    Example (sync)::

        with IncidentResponseClient(base_url="http://localhost:7860").sync() as client:
            result = client.reset()
            result = client.step(IncidentAction(action="block ip 203.0.113.45"))
    """

    ENV_NAME = "incident_response_env"
    Action = IncidentAction
    Observation = IncidentObservation
