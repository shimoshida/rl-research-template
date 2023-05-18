import torch
import pfrl


def main():
    # logger
    logger = None
    is_save = True
    if is_save:
        from mylib import WandbLogger

        # logger
        exp_name = "test"
        wandb_kwards = dict(
            project="test",
            name=exp_name,
            # save_code=True,
        )
        logger = WandbLogger(exp_name=exp_name, save_dir="outputs", **wandb_kwards)


if __name__ == "__main__":
    main()
