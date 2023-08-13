import copy
import json
import math
import os
from typing import Tuple, Sequence, Union, Dict, Optional, Any, cast, Generator, List

import cv2
import numpy as np
from PIL import Image, ImageDraw
from ai2thor.controller import Controller
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import colour as col

from allenact.utils.system import get_logger
from allenact.utils.viz_utils import TrajectoryViz

class ThorPositionTo2DFrameTranslator(object):
    def __init__(
        self,
        frame_shape_rows_cols: Tuple[int, int],
        cam_position: Sequence[float],
        orth_size: float,
    ):
        self.frame_shape = frame_shape_rows_cols
        self.lower_left = np.array((cam_position[0], cam_position[2])) - 1.3 * orth_size
        self.span = 2.6 * orth_size

    def __call__(self, position: Sequence[float]):
        if len(position) == 3:
            x, _, z = position
        else:
            x, z = position

        camera_position = (np.array((x, z)) - self.lower_left) / self.span
        return np.array(
            (
                round(self.frame_shape[0] * (1.0 - camera_position[1])),
                round(self.frame_shape[1] * camera_position[0]),
            ),
            dtype=int,
        )


class ThorViz(TrajectoryViz):
    def __init__(
        self,
        scenes: list,
        houses_dir: str,
        path_to_trajectory: Sequence[str] = ("task_info", "followed_path"),
        label: str = "thor_trajectory",
        figsize: Tuple[float, float] = (8, 4),  # width, height
        fontsize: float = 10,
        viz_rows_cols: Tuple[int, int] = (448, 448),
        single_color: bool = False,
        view_triangle_only_on_last: bool = True,
        disable_view_triangle: bool = False,
        line_opacity: float = 1.0,
        **kwargs
    ):
        super().__init__(
            path_to_trajectory=path_to_trajectory,
            label=label,
            figsize=figsize,
            fontsize=fontsize,
            **kwargs
        )

        self.scenes = scenes
        self.houses_dir = houses_dir

        self.viz_rows_cols = viz_rows_cols
        self.single_color = single_color
        self.view_triangle_only_on_last = view_triangle_only_on_last
        self.disable_view_triangle = disable_view_triangle
        self.line_opacity = line_opacity

        # Only needed for rendering
        self.map_data: Optional[Dict[str, Any]] = None
        self.thor_top_downs: Optional[Dict[str, np.ndarray]] = None

        self.controller: Optional[Controller] = None

    def init_top_down_render(self):
        self.map_data = self.get_translator()
        self.thor_top_downs = self.make_top_down_views()

        # No controller needed after this point
        if self.controller is not None:
            self.controller.stop()
            self.controller = None

    def load_house_data(self, scene_name) -> Dict:
        with open(os.path.join(self.houses_dir, scene_name + '.json')) as json_file:
            house_data = (json.load(json_file))

        return house_data

    def get_translator(self) -> Dict[str, Dict]:
        map_data = {}
        self.make_controller()

        for scene in self.scenes:
            house_data = self.load_house_data(scene)
            self.controller.reset(house_data)
            map_data_scene = self.get_agent_map_data()

            pos_translator = ThorPositionTo2DFrameTranslator(
                self.viz_rows_cols,
                self.position_to_tuple(map_data_scene['pose']["position"]),
                map_data_scene["orthographicSize"],
            )

            map_data_scene["pos_translator"] = pos_translator
            map_data[scene] = map_data_scene

        return map_data

    def make_top_down_views(self) -> Dict[str, np.ndarray]:
        top_downs = {}
        # self.make_controller()

        for scene in self.scenes:
            # house_data = self.load_house_data(scene)
            # self.controller.reset(scene=house_data)

            event = self.controller.step(
                action="AddThirdPartyCamera",
                **self.map_data[scene]['pose'],
                skyboxColor="white",
                raise_for_failure=True,
            )
            top_downs[scene] = event.third_party_camera_frames[-1]

        return top_downs

    def make_controller(self):
        if self.controller is None:
            self.controller = Controller(quality='Very High', 
                rotateStepDegrees=30.0,
                visibilityDistance = 1.0,
                snapToGrid=False,
                gridSize=0.25,
                agentMode="locobot",
                width=self.viz_rows_cols[1],
                height=self.viz_rows_cols[0],
                fieldOfView=90,
                renderDepthImage= False,
                makeAgentsVisible= False
                )
            
    def get_agent_map_data(self):
        event = self.controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
        pose = copy.deepcopy(event.metadata["actionReturn"])
        orthographic_size = pose["orthographicSize"]

        bounds = event.metadata["sceneBounds"]["size"]
        max_bound = max(bounds["x"], bounds["z"])

        pose["fieldOfView"] = 50
        pose["position"]["y"] += max_bound
        pose["orthographic"] = False
        pose["farClippingPlane"] = 50
        del pose["orthographicSize"]

        to_return = {
            "orthographicSize": orthographic_size,
            "pose": pose,
        }

        return to_return

    @staticmethod
    def position_to_tuple(position: Dict[str, float]) -> Tuple[float, float, float]:
        return position["x"], position["y"], position["z"]

    @staticmethod
    def add_lines_to_map(
        ps: Sequence[Any],
        frame: np.ndarray,
        pos_translator: ThorPositionTo2DFrameTranslator,
        opacity: float,
        color: Optional[Tuple[int, ...]] = None,
    ) -> np.ndarray:
        if len(ps) <= 1:
            return frame
        if color is None:
            color = (255, 0, 0)

        img1 = Image.fromarray(frame.astype("uint8"), "RGB").convert("RGBA")
        img2 = Image.new("RGBA", frame.shape[:-1])  # Use RGBA

        opacity = int(round(255 * opacity))  # Define transparency for the triangle.
        draw = ImageDraw.Draw(img2)
        for i in range(len(ps) - 1):
            draw.line(
                tuple(reversed(pos_translator(ps[i])))
                + tuple(reversed(pos_translator(ps[i + 1]))),
                fill=color + (opacity,),
                width=int(frame.shape[0] / 100),
            )

        img = Image.alpha_composite(img1, img2)
        return np.array(img.convert("RGB"))

    @staticmethod
    def add_line_to_map(
        p0: Any,
        p1: Any,
        frame: np.ndarray,
        pos_translator: ThorPositionTo2DFrameTranslator,
        opacity: float,
        color: Optional[Tuple[int, ...]] = None,
    ) -> np.ndarray:
        if p0 == p1:
            return frame
        if color is None:
            color = (255, 0, 0)

        img1 = Image.fromarray(frame.astype("uint8"), "RGB").convert("RGBA")
        img2 = Image.new("RGBA", frame.shape[:-1])  # Use RGBA

        opacity = int(round(255 * opacity))  # Define transparency for the triangle.
        draw = ImageDraw.Draw(img2)
        draw.line(
            tuple(reversed(pos_translator(p0))) + tuple(reversed(pos_translator(p1))),
            fill=color + (opacity,),
            width=int(frame.shape[0] / 100),
        )

        img = Image.alpha_composite(img1, img2)
        return np.array(img.convert("RGB"))

    @staticmethod
    def add_agent_view_triangle(
        position: Any,
        rotation: Dict[str, float],
        frame: np.ndarray,
        pos_translator: ThorPositionTo2DFrameTranslator,
        scale: float = 1.0,
        opacity: float = 0.1,
    ) -> np.ndarray:
        p0 = np.array((position[0], position[2]))
        p1 = copy.copy(p0)
        p2 = copy.copy(p0)

        theta = -2 * math.pi * (rotation["y"] / 360.0)
        rotation_mat = np.array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
        )
        offset1 = scale * np.array([-1 / 2.0, 1])
        offset2 = scale * np.array([1 / 2.0, 1])

        p1 += np.matmul(rotation_mat, offset1)
        p2 += np.matmul(rotation_mat, offset2)

        img1 = Image.fromarray(frame.astype("uint8"), "RGB").convert("RGBA")
        img2 = Image.new("RGBA", frame.shape[:-1])  # Use RGBA

        opacity = int(round(255 * opacity))  # Define transparency for the triangle.
        points = [tuple(reversed(pos_translator(p))) for p in [p0, p1, p2]]
        draw = ImageDraw.Draw(img2)
        draw.polygon(points, fill=(255, 255, 255, opacity))

        img = Image.alpha_composite(img1, img2)
        return np.array(img.convert("RGB"))

    @staticmethod
    def visualize_agent_path(
        positions: Sequence[Any],
        frame: np.ndarray,
        pos_translator: ThorPositionTo2DFrameTranslator,
        single_color: bool = False,
        view_triangle_only_on_last: bool = False,
        disable_view_triangle: bool = False,
        line_opacity: float = 1.0,
        trajectory_start_end_color_str: Tuple[str, str] = ("purple", "pink"),
    ) -> np.ndarray:
        if single_color:
            frame = ThorViz.add_lines_to_map(
                list(map(ThorViz.position_to_tuple, positions)),
                frame,
                pos_translator,
                line_opacity,
                tuple(
                    map(
                        lambda x: int(round(255 * x)),
                        col.Color(trajectory_start_end_color_str[0]).rgb,
                    )
                ),
            )
        else:
            if len(positions) > 1:
                colors = list(
                    col.Color(trajectory_start_end_color_str[0]).range_to(
                        col.Color(trajectory_start_end_color_str[1]), len(positions) - 1
                    )
                )
            for i in range(len(positions) - 1):
                frame = ThorViz.add_line_to_map(
                    ThorViz.position_to_tuple(positions[i]),
                    ThorViz.position_to_tuple(positions[i + 1]),
                    frame,
                    pos_translator,
                    opacity=line_opacity,
                    color=tuple(map(lambda x: int(round(255 * x)), colors[i].rgb)),
                )

        if view_triangle_only_on_last:
            positions = [positions[-1]]
        if disable_view_triangle:
            positions = []
        for position in positions:
            frame = ThorViz.add_agent_view_triangle(
                ThorViz.position_to_tuple(position),
                rotation=position["rotation"],
                frame=frame,
                pos_translator=pos_translator,
                opacity=0.05 + view_triangle_only_on_last * 0.2,
            )
        return frame

    def make_fig(self, episode: Any, episode_id: str) -> Figure:
        trajectory: Sequence[Dict[str, Any]] = self._access(
            episode, self.path_to_trajectory
        )

        if self.thor_top_downs is None:
            self.init_top_down_render()

        scene_name = "_".join(episode_id.split("_")[:4])

        im = self.visualize_agent_path(
            trajectory,
            self.thor_top_downs[scene_name],
            self.map_data[scene_name]["pos_translator"],
            single_color=self.single_color,
            view_triangle_only_on_last=self.view_triangle_only_on_last,
            disable_view_triangle=self.disable_view_triangle,
            line_opacity=self.line_opacity,
        )

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_title(episode_id, fontsize=self.fontsize)
        ax.imshow(im)
        ax.axis("off")

        return fig