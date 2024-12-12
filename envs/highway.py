from __future__ import annotations
import numpy as np
from highway_env import utils
from highway_env.envs import HighwayEnv
from highway_env.envs.common.action import Action
from highway_env.vehicle.controller import ControlledVehicle

Observation = np.ndarray

class Highway(HighwayEnv):
    @classmethod
    def config(cls, myconfig: dict) -> dict:
        config = super().default_config()
        config.update(myconfig)
        return config

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"] + self.config["acceleration_reward"] + self.config["lane_change_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: Action) -> dict[str, float]:
        # print("acceleration -> ", self.vehicle.action["acceleration"])
        # write acceleration to a file
        # with open("acceleration.txt", "a") as f:
        #     if not self.vehicle.crashed:
        #         f.write(str(min(self.vehicle.action["acceleration"], 0)) + "\n")
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
            "acceleration_reward": (min(self.vehicle.action["acceleration"], 0))**2,
        }

class HighwayFast(Highway):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "simulation_frequency": 5,
                "duration": 30,  # [s]
                "ego_spacing": 1.5,
            }
        )
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False
