import os
from datetime import datetime
from ulid import ULID


class AgentTrainer:
    """
    AgentTrainer can train and evaluate an agent for a reinforcement learning task.
    """

    def __init__(self, model_dir="", agent_id=ULID()):
        self.agent_id = agent_id
        self.model_dir = model_dir

    def train(self, model, sessions=40, runs_per_session=4, model_version="0.1.0"):
        """
        Train the agent for the specified number of steps.
        """

        model_name = model.__class__.__name__
        if self.model_dir:
            os.makedirs(self.model_dir, exist_ok=True)

        # Train for the number of sessions
        for _ in range(sessions):
            model.learn(total_timesteps=model.n_steps * runs_per_session)
            session_end = datetime.now()

            if self.model_dir:
                model.save(
                    os.path.join(
                        self.model_dir,
                        "{}_{}.zip".format(
                            model_name, session_end.strftime("%Y%m%d-%H%M%S")
                        ),
                    )
                )
