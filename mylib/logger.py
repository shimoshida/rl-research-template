import os
import wandb


class WandbLogger:
    def __init__(
        self,
        exp_name: str,
        offline: bool = False,
        save_dir: str = None,
        id: str = None,
        project: str = None,
        **kwargs,
    ) -> None:
        self.offline = offline
        self.save_dir = save_dir
        self.id = id
        self.project = project
        self._wandb_kwargs = {
            "name": exp_name,
            "dir": save_dir,
            "id": id,
            "project": project,
            "resume": "allow",
            **kwargs,
        }
        self._has_imported_wandb = False
        self.exp_name = exp_name
        self.log_dir = save_dir
        self.experiment = self._create_experiment()
        if self.offline:
            os.environ["WANDB_MODE"] = "dryrun"

    def _create_experiment(self) -> "WandbLogger":
        """Creates a wandb experiment.
        Args:
            exp_name (str): The name of the experiment.
        Returns:
            WandbLogger: The wandb experiment logger.
        """
        if self.offline:
            os.environ["WANDB_MODE"] = "dryrun"

        return wandb.init(**self._wandb_kwargs)

    def log_table(
        self, name: str, data: list[list[object]], columns: list[str]
    ) -> None:
        """"""
        table = wandb.Table(
            data=data,
            columns=columns,
        )
        self.experiment.log({name: table})

    def log_scalar(self, name: str, value: float, step: int = None) -> None:
        """Logs a scalar value to wandb.
        Args:
            name (str): The name of the scalar.
            value (float): The value of the scalar.
            step (int, optional): The step at which the scalar is logged.
                Defaults to None.
        """
        if step is not None:
            self.experiment.log({name: value, "trainer/step": step})
        else:
            self.experiment.log({name: value})

    def log_hparams(self, cfg: "DictConfig") -> None:  # noqa: F821
        """Logs the hyperparameters of the experiment.
        Args:
            cfg (DictConfig): The configuration of the experiment.
        """
        self.experiment.config.update(cfg, allow_val_change=True)

    def save_agent(self, agent, n_steps: int):
        """Save trained parameters of agents
        Args:
            agent (pfrl.Agents): agent.
            n_steps (int): timesteps.
        """
        dirname = os.path.join(wandb.run.dir, "trained-agents", f"{n_steps}ep-agent")
        agent.save(dirname)

    def __repr__(self) -> str:
        return f"WandbLogger(experiment={self.experiment.__repr__()})"
