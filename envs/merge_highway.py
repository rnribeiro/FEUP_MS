from __future__ import annotations
from highway_env import utils
from highway_env.envs import MergeEnv
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.utils import near_split
import tensorflow as tf
import numpy as np


class MergeHighway(MergeEnv):
    @classmethod
    def config(cls, myconfig: dict) -> dict:
        config = super().default_config()
        config.update(myconfig)
        return config


    def _make_vehicles(self) -> None:
        super()._make_vehicles()
        with open('action.txt', 'w') as f:
            f.write("")  

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        with open('action.txt', 'a') as f:
            f.write(str(action) + "\n")
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"] + self.config["lane_change_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"] + self.config["idle_reward"],
                ],
                [0, 1],
            )

        reward *= rewards["on_road_reward"]
        print("reward -> ", reward)
        return reward

    def _rewards(self, action: int) -> dict[str, float]:
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            self.vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )

        print("speed -> ", forward_speed)
        
        with open('action.txt', 'r') as f:
            lines = f.readlines()
            idle = sum([1 for line in lines if "1" in line]) / sum([1 for line in lines if line in ["1\n","3\n","4\n"]]) if sum([1 for line in lines if line in ["1\n","3\n","4\n"]]) > 0 else 0

        print("idle % -> ", idle)
        idle = idle if (idle < self.config["idle_percentage_range"][1] and idle > self.config["idle_percentage_range"][0] )else 0
        scaled_idle = utils.lmap(
            idle, self.config["idle_percentage_range"], [0, 1]
        )


        return {
            "collision_reward":float(self.vehicle.crashed),
            "right_lane_reward": self.vehicle.lane_index[2] / 1,
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "lane_change_reward": action in [0, 2],
            "on_road_reward": float(self.vehicle.on_road),
            "idle_reward": np.clip(scaled_idle, 0, 1),
        }
    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road

        # Place the ego vehicle on the merging lane
        ego_vehicle = self.action_type.vehicle_class(
            road, road.network.get_lane(("j", "k", 0)).position(30, 0), speed=20
        )
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        # Add other vehicles on the highway
        for position, speed in [(90, 29), (70, 31), (5, 31.5)]:
            lane = road.network.get_lane(("a", "b", self.np_random.integers(2)))
            position = lane.position(position + self.np_random.uniform(-5, 5), 0)
            speed += self.np_random.uniform(-1, 1)
            road.vehicles.append(other_vehicles_type(road, position, speed=speed))

        # Add a merging vehicle on the ramp
        merging_v = other_vehicles_type(
            road, road.network.get_lane(("j", "k", 0)).position(110, 0), speed=20
        )
        merging_v.target_speed = 30
        road.vehicles.append(merging_v)
        self.vehicle = ego_vehicle

    def _get_values(self):
        return self.vehicle.speed, self.vehicle.crashed