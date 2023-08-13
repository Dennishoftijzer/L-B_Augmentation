import glob
import json
from math import ceil
import os
from typing import Dict, Any, List, Optional, Sequence, Union
import platform

import gym
import ai2thor
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.base_abstractions.experiment_config import ExperimentConfig, MachineParams
from allenact.base_abstractions.preprocessor import Preprocessor, SensorPreprocessorGraph
from allenact.base_abstractions.sensor import ExpertActionSensor, SensorSuite
from allenact.base_abstractions.task import TaskSampler
from allenact.utils.experiment_utils import (
    Builder,
    PipelineStage,
    TrainingPipeline,
    LinearDecay,
    evenly_distribute_count_into_bins,
)
from allenact_plugins.ithor_plugin.ithor_sensors import (
    RGBSensorThor,
    GoalObjectTypeThorSensor,
)
from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_util import horizontal_to_vertical_fov
from allenact_plugins.procthor_plugin.procthor_task_samplers import ObjectNavTaskSampler
from allenact_plugins.procthor_plugin.procthor_tasks import ObjectNavTask
from projects.objectnav_baselines.experiments.clip.zeroshot_mixins import LB_AugResNetPreprocessGRUActorCriticMixin
from projects.objectnav_baselines.experiments.objectnav_base import ObjectNavBaseConfig


from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor
from projects.objectnav_baselines.experiments.clip.mixins import (
    ClipResNetPreprocessGRUActorCriticMixin,
)
from projects.objectnav_baselines.mixins import ObjectNavPPOMixin


class LBaug_EmbCLIP_ObjectNavProcThorPPOExperimentConfig(ObjectNavBaseConfig):
    """A simple object navigation experiment in ProcTHOR training with PPO."""

    CLIP_MODEL_TYPE = "RN50"

    OBJECT_TYPES = tuple(
        sorted(
            [                   # Target rooms:
                "AlarmClock",   # Bedroom
                "Bed",          # ,,
                "Dresser",      # ,,
                "Television",   # Livingroom
                "Sofa",         # ,,
                "Newspaper",    # ,,
                "Fridge",       # Kitchen
                "Kettle",       # ,,
                "Apple"         # ,,
            ],
        )
    )

    # Scenes to use during training:
    HOUSES_DATASET_DIR_TRAIN = os.path.join(os.getcwd(), "datasets/ProcTHOR/Train")
    TRAIN_SCENES= ["157_LR0_K1_BR2", 
                    "167_LR0_K1_BR2", 
                    "249_LR0_K1_BR2", 
                    "552_LR0_K1_BR2", 
                    "1430_LR0_K1_BR2",
                    "1500_LR0_K1_BR2",
                    "1713_LR0_K1_BR2",
                    "1917_LR0_K1_BR2",
                    "1965_LR0_K1_BR2",
                    "2102_LR0_K1_BR2",
                    "2107_LR0_K1_BR2",
                    "2950_LR0_K1_BR2",
                    "3249_LR0_K1_BR2",
                    "3395_LR0_K1_BR2",
                    "3440_LR0_K1_BR2",
                    "3553_LR0_K1_BR2", 
                    "4167_LR0_K1_BR2",
                    "4262_LR0_K1_BR2",
                    "4807_LR0_K1_BR2",
                    "5038_LR0_K1_BR2"
                    ]
    # and validation:
    HOUSES_DATASET_DIR_VAL = os.path.join(os.getcwd(), "datasets/ProcTHOR/Val")
    VALID_SCENES = ["5902_LR0_K1_BR2",
                    "5922_LR0_K1_BR2",
                    "6378_LR0_K1_BR2",
                    "6763_LR0_K1_BR2",
                    "6816_LR0_K1_BR2"
                    ]

    # We run seperate experiments for each target object (see scripts.py).
    TEST_OBJECT_TYPES = tuple(sorted(["Dresser",],))

    # Test scenes:
    HOUSES_DATASET_DIR_TEST= os.path.join(os.getcwd(), "datasets/ProcTHOR/Test/3_ALL")
    TEST_SCENES = [os.path.splitext(os.path.basename(scene))[0] for scene in glob.glob(os.path.join(HOUSES_DATASET_DIR_TEST, "*.json"))]

    # Setting up sensors and basic environment details
    SENSORS = [
        RGBSensorThor(
            height=ObjectNavBaseConfig.SCREEN_SIZE,
            width=ObjectNavBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
        ),
        GoalObjectTypeThorSensor(object_types=OBJECT_TYPES)
    ]
    
    ENV_ARGS = {
        "height": ObjectNavBaseConfig.SCREEN_SIZE,
        "width": ObjectNavBaseConfig.SCREEN_SIZE,
        "quality": "Very Low",
        # 'continuousMode': True,  
        'rotateStepDegrees': 30.0,
        'visibilityDistance': 1.0,
        'gridSize': 0.25, 
        'snapToGrid': False,
        "agentMode": "locobot",
        'renderDepthImage': False,
        'fov': 90
    }

    NUM_PROCESSES = 40                                          # I recommend at least 20, such that each tasksampler initilizes a single scene (resetting into new scenes massively slow down training).
    DEFAULT_TRAIN_GPU_IDS: Sequence[int] = list(range(torch.cuda.device_count()))
    DEFAULT_VALID_GPU_IDS: Sequence[int] = [torch.cuda.device_count() - 1]
    DEFAULT_TEST_GPU_IDS: Sequence[int] = [torch.cuda.device_count() - 1]
    VALID_SAMPLES_IN_SCENE = 40
    TEST_SAMPLES_IN_SCENE = 3

    ACTION_SPACE = gym.spaces.Discrete(len(ObjectNavTask.class_action_names()))

    def __init__(self, add_prev_actions: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.preprocessing_and_model = LB_AugResNetPreprocessGRUActorCriticMixin(
            sensors=self.SENSORS,
            clip_model_type=self.CLIP_MODEL_TYPE,
            screen_size=ObjectNavBaseConfig.SCREEN_SIZE,
            goal_sensor_type=GoalObjectTypeThorSensor,
            pool=True,
            pooling_type='attn',
            target_types=self.OBJECT_TYPES
        )
        self.add_prev_actions = add_prev_actions

    
    @classmethod
    def tag(cls):
        return "LBaug-ObjectNav-ProcTHOR-RGB-ClipResNet50GRU-DDPPO"

    @classmethod
    def training_pipeline(self, **kwargs) -> TrainingPipeline:
        return ObjectNavPPOMixin.training_pipeline(
            auxiliary_uuids=[],
            multiple_beliefs=False,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD, # no. of rollouts before enforcing scene advance in all samplers
            ppo_steps = int(30e6),
            save_interval= 200000,
            log_interval=10000,
            gamma=0.99,                     # Discount factor for reward
            gae_lambda=0.95,                # generalized advantage estimation (GAE) parameter
            # From PPOconfig class:
            # value loss coef       = 0.5  
            # entropy_coef          = 0.01
            # clip_param (epsilon)  = 0.1
            num_steps=192,                  # Total no. of steps a single agent takes in a rollout
            num_mini_batch=1,               # No. of minibatches per rollout
            lr=3e-4,                        # Learning rate
            # Optimizer             = Adam
            max_grad_norm=0.5,              # Gradient clip norm
            # update_repeats        = 4     # No. of times we will cycle through the mini-batches corresponding to a single rollout doing gradient updates
        )

    
    def preprocessors(self) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        return self.preprocessing_and_model.preprocessors()

    def create_model(self, **kwargs) -> nn.Module:
        return self.preprocessing_and_model.create_model(
            num_actions=self.ACTION_SPACE.n,
            add_prev_actions=self.add_prev_actions,
            **kwargs
        )

    def machine_params(self, mode="train", **kwargs):
        sampler_devices: List[int] = []
        if mode == "train":
            workers_per_device = 1
            gpu_ids = (
                []
                if not torch.cuda.is_available()
                else list(self.DEFAULT_TRAIN_GPU_IDS) * workers_per_device
            )
            nprocesses = (
                8
                if not torch.cuda.is_available()
                else evenly_distribute_count_into_bins(self.NUM_PROCESSES, len(gpu_ids))
            )
            sampler_devices = list(self.DEFAULT_TRAIN_GPU_IDS)
        elif mode == "valid":
            nprocesses = len(self.VALID_SCENES)
            gpu_ids = [] if not torch.cuda.is_available() else self.DEFAULT_VALID_GPU_IDS
        elif mode == "test":
            workers_per_device = 1
            gpu_ids = (
                []
                if not torch.cuda.is_available()
                else list(self.DEFAULT_TEST_GPU_IDS) * workers_per_device
            )
            nprocesses = (
                evenly_distribute_count_into_bins(len(self.TEST_SCENES), len(gpu_ids))
            )
            sampler_devices = list(self.DEFAULT_TEST_GPU_IDS)
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        sensors = [*self.SENSORS]
        if mode != "train":
            sensors = [s for s in sensors if not isinstance(s, ExpertActionSensor)]

        sensor_preprocessor_graph = (
            SensorPreprocessorGraph(
                source_observation_spaces=SensorSuite(sensors).observation_spaces,
                preprocessors=self.preprocessors(),
            )
            if mode == "train"
            or (
                (isinstance(nprocesses, int) and nprocesses > 0)
                or (isinstance(nprocesses, Sequence) and sum(nprocesses) > 0)
            )
            else None
        )

        return MachineParams(
            nprocesses=nprocesses,
            devices=gpu_ids,
            sampler_devices=sampler_devices
            if mode == "train"
            else gpu_ids,  # ignored with > 1 gpu_ids
            sensor_preprocessor_graph=sensor_preprocessor_graph,
        )


    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return ObjectNavTaskSampler(**kwargs)

    @staticmethod
    def _partition_inds(n: int, num_parts: int):
        return np.round(np.linspace(0, n, num_parts + 1, endpoint=True)).astype(
            np.int32
        )

    def _get_sampler_args_for_scene_split(
        self,
        scenes: List[str],
        process_ind: int,
        total_processes: int,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        if total_processes > len(scenes):  # oversample some scenes -> bias
            if total_processes % len(scenes) != 0:
                print(
                    "Warning: oversampling some of the scenes to feed all processes."
                    " You can avoid this by setting a number of workers divisible by the number of scenes"
                )
            scenes = scenes * int(ceil(total_processes / len(scenes)))
            scenes = scenes[: total_processes * (len(scenes) // total_processes)]
        else:
            if len(scenes) % total_processes != 0:
                print(
                    "Warning: oversampling some of the scenes to feed all processes."
                    " You can avoid this by setting a number of workers divisor of the number of scenes"
                )
        inds = self._partition_inds(len(scenes), total_processes)

        return {
            "scenes": scenes[inds[process_ind] : inds[process_ind + 1]],
            "object_types": self.OBJECT_TYPES,
            "env_args": self.ENV_ARGS,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(
                len(ObjectNavTask.class_action_names())
            ),
            "seed": seeds[process_ind] if seeds is not None else None,
            "deterministic_cudnn": deterministic_cudnn,
            "rewards_config": self.REWARD_CONFIG,
        }

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            self.TRAIN_SCENES,
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        res["scene_period"] = "manual"
        res["env_args"] = {}
        res["env_args"].update(self.ENV_ARGS)
        res["env_args"].update({"houses_dir": self.HOUSES_DATASET_DIR_TRAIN}),
        res["env_args"]["x_display"] = (
            ("0.%d" % devices[process_ind % len(devices)])
            if devices is not None and len(devices) > 0
            else None
        )
        return res

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            self.VALID_SCENES,
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        res["scene_period"] = self.VALID_SAMPLES_IN_SCENE
        res["max_tasks"] = self.VALID_SAMPLES_IN_SCENE * len(res["scenes"])
        res["env_args"] = {}
        res["env_args"].update(self.ENV_ARGS)
        res["env_args"].update({"houses_dir": self.HOUSES_DATASET_DIR_VAL}),
        res["env_args"]["x_display"] = (
            ("0.%d" % devices[process_ind % len(devices)])
            if devices is not None and len(devices) > 0
            else None
        )
        return res

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:

        res = self._get_sampler_args_for_scene_split(
            self.TEST_SCENES,
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )

        res["scene_period"] = self.TEST_SAMPLES_IN_SCENE
        res["max_tasks"] = self.TEST_SAMPLES_IN_SCENE * len(res["scenes"])
        res["env_args"] = {}
        res["env_args"].update(self.ENV_ARGS)
        res["object_types"] = (self.TEST_OBJECT_TYPES)
        res["env_args"].update({"houses_dir": self.HOUSES_DATASET_DIR_TEST}),
        res["env_args"]["x_display"] = (
            ("0.%d" % devices[process_ind % len(devices)])
            if devices is not None and len(devices) > 0
            else None
        )
        return res