import json
import asyncio
import logging

import numpy as np
from pyensign.events import Event
from stable_baselines3.common.logger import KVWriter, filter_excluded_keys

class EnsignWriter(KVWriter):
    """
    EnsignWriter subclasses the Stable Baselines3 KVWriter class to write key-value pairs to Ensign.
    """

    def __init__(self, ensign, topic="agent-training", agent_id=None):
        super().__init__()
        self.ensign = ensign
        self.topic = topic
        self.version = "0.1.0"
        self.agent_id = agent_id

    async def publish(self, event):
        """
        One-off publish to Ensign.
        """

        await self.ensign.publish(self.topic, event)
        try:
            await self.ensign.flush()
        except asyncio.TimeoutError:
            logging.warning("Timeout exceeded while flushing Ensign writer.")

    def write(self, key_values, key_excluded, step=0):
        """
        Write the key-value pairs to Ensign.
        """

        meta = {"step": step}
        if self.agent_id:
            meta["agent_id"] = str(self.agent_id)
        key_values = filter_excluded_keys(key_values, key_excluded, "ensign")
        for key, value in key_values.items():
            # JSON doesn't support numpy types
            if isinstance(value, np.float32):
                key_values[key] = float(value)

        event = Event(json.dumps(key_values).encode("utf-8"), mimetype="application/json", schema_name="training_log", schema_version=self.version, meta={"agent_id": str(self.agent_id), "step_number": str(step)})
   
        publish = self.publish(event)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            asyncio.run(publish)
            return

        asyncio.run_coroutine_threadsafe(publish, loop)

    def close(self):
        """
        Close the Ensign writer.
        """

        asyncio.run(self.ensign.close())