import argparse
import glob
import gzip
import json
import os
import copy

from typing import Tuple, Dict, List, Any
from projects.objectnav_baselines.experiments.procthor.CW_objectnav_procthor_rgb_clipresnet50gru_ddppo import CW_EmbCLIP_ObjectNavProcThorPPOExperimentConfig

from allenact_plugins.ithor_plugin.ithor_util import round_to_factor

def get_argument_parser():
    """Creates the argument parser for easily running all experiments without all config arguments"""

    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        description="Script to generate episodes for visualization. \n\n" 
        "The script will generate n episodes for each possible target object, depending on a specified directory and target room.\n" 
        "For example, it will generate n episodes for each target object in [\"Television\",\"Sofa\",\"Newspaper\"] for each scene in \"datasets/ProcTHOR/Test/1_livingroom\". \n"
        "Possible target objects: \n"
        "Living room: [\"Television\",\"Sofa\",\"Newspaper\",]\n" 
        "Bedroom    : [\"AlarmClock\", \"Bed\", \"Dresser\"]\n"  
        "Kitchen    : [\"Fridge\", \"Kettle\", \"Apple\"]\n\n"
        "Optionally you can provide: \n"
        "-dir   Path to scene directory to use for episode generation       (default: datasets/ProcTHOR/Test/0_ALL). \n"
        "-s     Subset of scenes within that directory                      (default: all scenes within dir). \n"
        "-n     Number of episodes to generate per target object per scene. (default: 1). \n",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "-dir",
        "--directory",
        required=False,
        default="datasets/ProcTHOR/Test/0_ALL",
        type=str,
        help="Specify path to one of the scene directories in the dataset e.g., datasets/ProcTHOR/Test/1_livingroom.",
    )

    parser.add_argument(
        "-s",
        "--scenes",
        required=False,
        type=str,
        nargs='*',
        help="Optionally, you can provide the names of a subset of scenes in the directory e.g., \"5902_LR0_K1_BR2\"",
    )

    parser.add_argument(
        "-n",
        "--numberofeps",
        required=False,
        default=1,
        type=int,
        help="Number of episodes to generate per scene per target object.",
    )

    return parser


def get_key(input_dict: Dict[str, Any]) -> Tuple[float, float, int, int]:
    if "x" in input_dict:
        x = input_dict["x"]
        z = input_dict["z"]
        orient = input_dict["initial_orientation"]
    return (
        round(x, 2),
        round(z, 2),
        round_to_factor(orient, 30) % 360,
    )

def generate_episodes_from_scenes(
    scenes: List[str],
    target_object_types: List[str],
    base_dir_output_path: str,
    num_tasks: int,
    # target_room: str,
):

    episodes_dir = os.path.join(base_dir_output_path, f"episodes_vizualization")
    os.makedirs(episodes_dir, exist_ok=True)

    exp_config = CW_EmbCLIP_ObjectNavProcThorPPOExperimentConfig()

    for scene in scenes:
        task_sampler_kwargs = exp_config.train_task_sampler_args(
        process_ind=0,
        total_processes= 1
        )

        task_sampler_kwargs['scenes'] = [scene]
        env_args = task_sampler_kwargs['env_args']

        env_args['houses_dir'] = base_dir_output_path
        task_sampler_kwargs['env_args'].update(env_args)

        episodes = []

        task_sampler = exp_config.make_sampler_fn(**task_sampler_kwargs)
        task_sampler_kwargs['max_tasks'] = num_tasks*len(target_object_types)

        for target_object in target_object_types:
            # task_sampler_kwargs['object_types'] = [target_object]
            task_sampler.object_types = [target_object]

            for i in range(num_tasks):
                task = task_sampler.next_task()

                task_info = copy.deepcopy(task.task_info)
                task_info.pop('mirrored')   
                task_info.pop('followed_path')
                task_info.pop('taken_actions')
                task_info.pop('action_names')

                pose = copy.deepcopy(task_info['initial_position'])
                pose['initial_orientation'] = copy.deepcopy(task_info['initial_orientation'])

                task_info["id"] = f"{task_info['scene']}__{'_'.join(list(map(str, get_key(pose))))}__{task_info['object_type']}"

                print(f"Generated episode {i+1}/{num_tasks} in scene: {scene} with target: {target_object}: \n {task_info}")
                episodes.append(task_info)

        task.close()

        episodes_file = os.path.join(episodes_dir, "episodes_" + scene + ".json")

        with open(episodes_file, "w") as outfile:
            json.dump(episodes, outfile)

        # json_str = json.dumps(episodes)
        # json_bytes = json_str.encode("utf-8")
        # with gzip.GzipFile(episodes_file, "w") as fout:
        #     fout.write(json_bytes)
        assert os.path.exists(episodes_file)
    
def main():
    parser = get_argument_parser()
    args = parser.parse_args()

    if args.scenes == None:
        SCENES = sorted([os.path.splitext(os.path.basename(scene))[0] for scene in glob.glob(os.path.join(args.directory, "*.json"))])
    else:
        SCENES = args.scenes

    TARGETS = {
                'ALL'           : ['Television','Sofa','Newspaper', 'Fridge','Kettle','Apple', 'AlarmClock','Bed', 'Dresser'],
                'Livingroom'    : ['Television','Sofa','Newspaper'],
                'Kitchen'       : ['Fridge','Kettle','Apple'],
                'Bedroom'       : ['AlarmClock','Bed', 'Dresser'] 
            }

    
    target_room = os.path.basename(args.directory).split("_")[1]

    targets = TARGETS[target_room]

    generate_episodes_from_scenes(scenes=SCENES, target_object_types=targets, base_dir_output_path=args.directory, num_tasks = args.numberofeps)

if __name__ == "__main__":    
    main()