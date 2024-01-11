import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        prog="JAXIRL",
    )

    parser.add_argument(
        "-loss",
        "--loss",
        default="IRL",
        type=str,
        choices=["IRL", "BC", "NONE"],
    )
    parser.add_argument(
        "-e",
        "--env",
        choices=[
            "halfcheetah",
            "ant",
            "humanoid",
            "hopper",
            "walker2d",
            "Reacher-misc",
            "CartPole-v1",
            "Pendulum-v1",
        ],
    )
    parser.add_argument("-l", "--log", action="store_true")
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
    )
    parser.add_argument(
        "-g",
        "--generations",
        default=100,
        type=int,
    )
    parser.add_argument(
        "-r",
        "--reward_type",
        choices=[
            "REWARD_STATE",
            "REWARD_STATE_ACTION",
            "NONE",
        ],
    )
    parser.add_argument("-sd", "--seed", nargs="*", default=[1])
    parser.add_argument("--run_test", action="store_true")
    parser.add_argument("--seeds", default=3, type=int)

    return parser
