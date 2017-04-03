import pprint
import random
import time

import numpy as np
import pandas as pd
from .entities import Snake, Field, CellType, SnakeAction, ALL_SNAKE_ACTIONS


class Environment(object):

    def __init__(self, config, debug=False):
        self.field = Field(level_map=config['field'])
        self.snake = None
        self.fruit = None
        self.rewards = config['rewards']
        self.max_step_limit = config.get('max_step_limit', 1000)
        self.is_game_over = False

        self.timestep_index = 0
        self.current_action = None
        self.stats = EpisodeStatistics()
        self.debug = debug
        self.debug_file = None
        self.stats_file = None

    def seed(self, value):
        random.seed(value)
        np.random.seed(value)

    def new_episode(self):
        self.field.create_level()
        self.stats.reset()
        self.timestep_index = 0

        self.snake = Snake(self.field.find_snake_head())
        self.field.place_snake(self.snake)
        self.generate_fruit()
        self.current_action = None
        self.is_game_over = False

        result = TimestepResult(
            observation=self.get_observation(),
            reward=0,
            is_episode_end=self.is_game_over
        )

        self.record_timestep_stats(result)
        return result

    def record_timestep_stats(self, result):
        if self.debug and self.debug_file is None:
            timestamp = time.strftime('%Y%m%d-%H%M%S')
            self.debug_file = open(f'snake-env-{timestamp}.log', 'w')
            # Write CSV header only.
            self.stats_file = open(f'snake-env-{timestamp}.csv', 'w')
            stats_csv_header_line = self.stats.to_dataframe()[:0].to_csv(index=None)
            print(stats_csv_header_line, file=self.stats_file, end='', flush=True)

        self.stats.record_timestep(self.current_action, result)
        self.stats.timesteps_survived = self.timestep_index

        if self.debug:
            print(result, file=self.debug_file)
            if result.is_episode_end:
                print(self.stats, file=self.debug_file)
                stats_csv_line = self.stats.to_dataframe().to_csv(header=False, index=None)
                print(stats_csv_line, file=self.stats_file, end='', flush=True)

    def get_observation(self):
        return np.copy(self.field._cells)

    def choose_action(self, action):
        self.current_action = action
        if action == SnakeAction.TURN_LEFT:
            self.snake.turn_left()
        elif action == SnakeAction.TURN_RIGHT:
            self.snake.turn_right()

    def timestep(self):
        self.timestep_index += 1
        reward = 0

        old_head = self.snake.head
        old_tail = self.snake.tail

        # Are we about to eat the fruit?
        if self.snake.peek_next_move() == self.fruit:
            self.snake.grow()
            self.generate_fruit()
            old_tail = None
            reward += self.rewards['ate_fruit']
            self.stats.fruits_eaten += 1

        # If not, just move forward.
        else:
            self.snake.move()
            reward += self.rewards['timestep']

        self.field.update_snake_footprint(old_head, old_tail, self.snake.head)

        # Hit a wall or own body?
        if not self.is_alive():
            if self.has_hit_wall():
                self.stats.termination_reason = 'hit_wall'
            if self.has_hit_own_body():
                self.stats.termination_reason = 'hit_own_body'

            self.field[self.snake.head] = CellType.SNAKE_HEAD
            self.is_game_over = True
            reward = self.rewards['died']

        # Exceeded the limit of moves?
        if self.timestep_index >= self.max_step_limit:
            self.is_game_over = True
            self.stats.termination_reason = 'timestep_limit_exceeded'

        result = TimestepResult(
            observation=self.get_observation(),
            reward=reward,
            is_episode_end=self.is_game_over
        )

        self.record_timestep_stats(result)
        return result

    def generate_fruit(self, position=None):
        if position is None:
            position = self.field.get_random_empty_cell()
        self.field[position] = CellType.FRUIT
        self.fruit = position

    def has_hit_wall(self):
        return self.field[self.snake.head] == CellType.WALL

    def has_hit_own_body(self):
        return self.field[self.snake.head] == CellType.SNAKE_BODY

    def is_alive(self):
        return not self.has_hit_wall() and not self.has_hit_own_body()


class TimestepResult(object):
    def __init__(self, observation, reward, is_episode_end):
        self.observation = observation
        self.reward = reward
        self.is_episode_end = is_episode_end

    def __str__(self):
        field_map = '\n'.join([
            ''.join(str(cell) for cell in row)
            for row in self.observation
        ])
        return '{}\nR = {}   end={}\n'.format(field_map, self.reward, self.is_episode_end)


class EpisodeStatistics():
    def __init__(self):
        self.reset()

    def reset(self):
        self.timesteps_survived = 0
        self.sum_episode_rewards = 0
        self.fruits_eaten = 0
        self.termination_reason = None
        self.action_counter = {
            action: 0
            for action in ALL_SNAKE_ACTIONS
        }

    def record_timestep(self, action, result):
        self.sum_episode_rewards += result.reward
        if action is not None:
            self.action_counter[action] += 1

    def flatten(self):
        flat_stats = {
            'timesteps_survived': self.timesteps_survived,
            'sum_episode_rewards': self.sum_episode_rewards,
            'mean_reward': self.sum_episode_rewards / self.timesteps_survived if self.timesteps_survived else None,
            'fruits_eaten': self.fruits_eaten,
            'termination_reason': self.termination_reason,
        }
        flat_stats.update({
            f'action_counter_{action}': self.action_counter.get(action, 0)
            for action in ALL_SNAKE_ACTIONS
        })
        return flat_stats

    def to_dataframe(self):
        return pd.DataFrame([self.flatten()])

    def __str__(self):
        return pprint.pformat(self.flatten())
