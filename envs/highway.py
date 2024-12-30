from __future__ import annotations
import numpy as np
from highway_env import utils
from highway_env.envs import HighwayEnv
from highway_env.envs.common.action import Action
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.utils import near_split


Observation = np.ndarray

class Highway(HighwayEnv):
    @classmethod
    def config(cls, myconfig: dict) -> dict:
        config = super().default_config()
        config.update(myconfig)
        return config

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

        with open('action.txt', 'w') as f:
            f.write("")  


    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        # write action to txt file
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


    def _rewards(self, action: Action) -> dict[str, float]:
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
        print("speed -> ", forward_speed)

        # Get percentage of idle actions from action.txt
        with open('action.txt', 'r') as f:
            lines = f.readlines()
            idle = sum([1 for line in lines if "1" in line]) / sum([1 for line in lines if line in ["1\n","3\n","4\n"]]) if sum([1 for line in lines if line in ["1\n","3\n","4\n"]]) > 0 else 0
       
        print("idle % -> ", idle)
        idle = idle if (idle < self.config["idle_percentage_range"][1] and idle > self.config["idle_percentage_range"][0] )else 0
        scaled_idle = utils.lmap(
            idle, self.config["idle_percentage_range"], [0, 1]
        )

        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
            "idle_reward": np.clip(scaled_idle, 0, 1),
            "lane_change_reward": action in [0, 2],
        }

class HighwayFast(Highway):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "simulation_frequency": 5,
                "lanes_count": 3,
                "vehicles_count": 20,
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
