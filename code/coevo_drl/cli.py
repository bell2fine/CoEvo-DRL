import argparse
import json
import logging

import numpy as np

from .training import train


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(prog="coevo-drl")
    subcommands = command.add_subparsers(dest="command", required=True)
    training = subcommands.add_parser("train")
    training.add_argument("--steps", type=int, default=1_000_000)
    training.add_argument("--rollout-steps", type=int, default=2048)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--device", default="cuda")
    training.add_argument("--output", default="coevo_drl.pt")
    simulation = subcommands.add_parser("simulate")
    simulation.add_argument("--seed", type=int, default=42)
    return command


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    arguments = parser().parse_args()
    if arguments.command == "train":
        agent = train(arguments.steps, arguments.rollout_steps, arguments.seed, arguments.device)
        agent.save(arguments.output, arguments.seed, arguments.steps)
        return
    rng = np.random.default_rng(arguments.seed)
    result = {"seed": arguments.seed, "initialization_checksum": float(rng.random(100).sum())}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
