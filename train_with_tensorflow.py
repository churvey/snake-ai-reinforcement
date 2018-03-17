#!/usr/bin/env python3.6

""" Front-end script for training a Snake agent. """

import json
import time
import sys

import tensorflow as tf

from snakeai.agent import DeepQNetworkAgent
from snakeai.gameplay.environment import Environment
from snakeai.utils.cli import HelpOnFailArgumentParser
from snakeai.model.model import Model


def parse_command_line_args(args):
    """ Parse command-line arguments and organize them into a single structured object. """

    parser = HelpOnFailArgumentParser(
        description='Snake AI training client.',
        epilog='Example: train.py --level 10x10.json --num-episodes 30000'
    )

    parser.add_argument(
        '--level',
        required=True,
        type=str,
        help='JSON file containing a level definition.',
    )
    parser.add_argument(
        '--num-episodes',
        required=True,
        type=int,
        default=30000,
        help='The number of episodes to run consecutively.',
    )

    return parser.parse_args(args)


def create_snake_environment(level_filename):
    """ Create a new Snake environment from the config file. """

    with open(level_filename) as cfg:
        env_config = json.load(cfg)

    return Environment(config=env_config, verbose=1)


def create_dqn_model(env, num_last_frames):
    """
    Build a new DQN model to be used for training.

    Args:
        env: an instance of Snake environment.
        num_last_frames: the number of last frames the agent considers as state.

    Returns:
        A compiled DQN model.
    """

    inputs = tf.placeholder(tf.float32, (None, num_last_frames) + env.observation_shape, name="inputs")

    from tensorflow.python.client import device_lib

    local_device_protos = device_lib.list_local_devices()
    num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
    data_format = 'channels_last'

    if num_gpus > 0:
        data_format = "channels_first"
        network = inputs
    else:
        network = tf.transpose(inputs, perm=[0, 2, 3, 1])

    network = tf.layers.conv2d(inputs=network, filters=16,
                               kernel_size=(3, 3), data_format=data_format,
                               name="conv2d_1", activation=tf.nn.relu)
    network = tf.layers.conv2d(inputs=network, filters=32,
                               kernel_size=(3, 3), data_format=data_format,
                               name="conv2d_2",
                               activation=tf.nn.relu)
    network = tf.layers.flatten(inputs=network, name="flatten")
    network = tf.layers.dense(inputs=network, units=256, activation=tf.nn.relu, name="dense_1")
    network = tf.layers.dense(inputs=network, units=env.num_actions, name="output")

    return Model(inputs, network)


def main():
    parsed_args = parse_command_line_args(sys.argv[1:])

    env = create_snake_environment(parsed_args.level)
    model = create_dqn_model(env, num_last_frames=4)

    agent = DeepQNetworkAgent(
        model=model,
        memory_size=-1,
        num_last_frames=model.input_shape[1]
    )
    agent.train(
        env,
        batch_size=64,
        num_episodes=parsed_args.num_episodes,
        checkpoint_freq=parsed_args.num_episodes // 10,
        discount_factor=0.95
    )


if __name__ == '__main__':
    t0 = time.time()
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
    print(f'eslaped time:', time.time() - t0)
