import re
import os
import argparse 

def get_argument_parser():
    """Creates the argument parser for easily running all experiments without all config arguments"""

    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        description="Simple script to run all experiments in our paper: Language-Based Augmentation to Address Shortcut Learning in Object-Goal Navigation.\n\n"
        "Note that, for testing, we need to select different OOD scenes (different wall color permutation but same layout) for different target rooms, and thereby target objects. Therefore, we run a seperate experiment for each target object in the Allenact framework. Furthermore, this ensures we evenly distribute evaluation episodes over all target objects.\n\n"
        "TRAINING: \n"    
        "Dataset path:                  \"datasets/ProcTHOR/Train\"\n"
        "Results:                       \"experiment_output/objectnav_procthor\". \n\n"
        "TESTING: \n"
        "Default checkpoints path:      \"pretrained_model_ckpts/OOD_generalization/{closed_world_embclip,open_world_embclip,LBaug_embclip}\" \n"
        "Dataset path:                  \"datasets/ProcTHOR/Test\"\n"
        "Results:                       \"eval_output/objectnav_procthor\". Note that, the resulting directory layout will seperate each target object for each number of wall color changes.\n\n"
        "VIZUALIZATION: \n"
        "Results:                       \"viz_output/objectnav_procthor\".\n"
        "See (\"projects/objectnav_baselines/experiments/procthor/viz_objectnav_procthor_rgb_clipresnet50gru_ddppo.py\") on how to visualize specific episodes.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "experiment",
        choices=['closed_world_embclip', 'open_world_embclip', 'LBaug_embclip'],
        help="Choose an experiment to run. Experiment configs are located in: \"projects/objectnav_baselines/experiments/procthor\"",
    )

    parser.add_argument(
        "-b",
        "--experiment_base",
        required=False,
        default="projects/objectnav_baselines/experiments/procthor",
        type=str,
        help="Base directory for all experiment configurations (default: \"projects/objectnav_baselines/experiments/procthor\").",
    )

    parser.add_argument(
        "--train_output_dir",
        required=False,
        default="experiment_output/objectnav_procthor",
        type=str,
        help="Training results output folder (default: \"experiment_output/objectnav_procthor\").",
    )

    parser.add_argument(
        "--test_output_dir",
        required=False,
        default="eval_output/objectnav_procthor",
        type=str,
        help="Inference results output folder (default: \"eval_output/objectnav_procthor\").",
    )

    parser.add_argument(
        "--viz_output_dir",
        required=False,
        default="viz_output/objectnav_procthor",
        type=str,
        help="Vizualization results output folder (default: \"viz_output/objectnav_procthor\").",
    )

    parser.add_argument(
        "-c",
        "--checkpoint",
        required=False,
        default=None,
        type=str,
        help="Optional checkpoint file name to run testing with. When testing (see the `--eval` flag) specify the path to a particular individual checkpoint file (.pt)",
    )

    group = parser.add_mutually_exclusive_group()
 
    group.add_argument(
        "--eval",
        dest="eval",
        action="store_true",
        required=False,
        help="if you pass the `--eval` flag, run inference on the 5 OOD scenes (\"datasets/ProcTHOR/Test\").",
    )

    group.add_argument(
        "--viz",
        dest="viz",
        action="store_true",
        required=False,
        help="if you pass the `--viz` flag, it will run evaluation and generate videos of agent's RGB egocentric views.", 
    )

    parser.set_defaults(eval=False)

    return parser

def set_tag(exp_conf_path, new_tag) -> None:
    """
    Set the tag in an experiment config file.
    """
    # Read experiment config
    with open(exp_conf_path, 'r') as f:
        exp_conf_data = f.readlines()

    tag_line_ind = [exp_conf_data.index(line) for line in exp_conf_data if re.match("\s*def\stag\(cls\):" ,line)][0]+1

    exp_conf_data[tag_line_ind] = re.sub('([0-9]?_?[A-Za-z]*-)*ObjectNav-ProcTHOR-RGB-ClipResNet50GRU-DDPPO',
                                        new_tag,
                                        exp_conf_data[tag_line_ind])
    
    with open(exp_conf_path, 'w') as f:
        f.writelines(exp_conf_data)

def train(args) -> None: 
    if args.experiment == 'closed_world_embclip':
        exp_config_path = f'{args.experiment_base}/CW_objectnav_procthor_rgb_clipresnet50gru_ddppo.py'
        set_tag(exp_config_path, "CW-ObjectNav-ProcTHOR-RGB-ClipResNet50GRU-DDPPO")
    elif args.experiment == 'open_world_embclip':
        exp_config_path = f'{args.experiment_base}/OW_objectnav_procthor_rgb_clipresnet50gru_ddppo.py'
        set_tag(exp_config_path, "OW-ObjectNav-ProcTHOR-RGB-ClipResNet50GRU-DDPPO")
    elif args.experiment == 'LBaug_embclip':
        exp_config_path = f'{args.experiment_base}/LBaug_objectnav_procthor_rgb_clipresnet50gru_ddppo.py'
        set_tag(exp_config_path, "LBaug-ObjectNav-ProcTHOR-RGB-ClipResNet50GRU-DDPPO")

    cmd = f"PYTHONPATH=. python allenact/main.py {os.path.basename(exp_config_path)} -o {args.train_output_dir} -b {args.experiment_base}"
    os.system(cmd)
    return

def test(args) -> None:
    if args.experiment == 'closed_world_embclip':
        exp_config_path = f'{args.experiment_base}/CW_objectnav_procthor_rgb_clipresnet50gru_ddppo.py'
        ckpt_path       = r"pretrained_model_ckpts/OOD_generalization/closed_world_embclip/exp_ObjectNav-ProcTHOR-RGB-ClipResNet50GRU-DDPPO__stage_00__steps_000019906560.pt" if args.checkpoint is None else args.checkpoint
        base_tag        = "CW-ObjectNav-ProcTHOR-RGB-ClipResNet50GRU-DDPPO"
    elif args.experiment == 'open_world_embclip':
        exp_config_path = f'{args.experiment_base}/OW_objectnav_procthor_rgb_clipresnet50gru_ddppo.py'
        ckpt_path       = r"pretrained_model_ckpts/OOD_generalization/open_world_embclip/exp_open_world-ObjectNav-ProcTHOR-RGB-ClipResNet50GRU-DDPPO__stage_00__steps_000029237760.pt" if args.checkpoint is None else args.checkpoint
        base_tag        = "OW-ObjectNav-ProcTHOR-RGB-ClipResNet50GRU-DDPPO"
    elif args.experiment == 'LBaug_embclip':
        exp_config_path = f'{args.experiment_base}/LBaug_objectnav_procthor_rgb_clipresnet50gru_ddppo.py'
        ckpt_path       = r"pretrained_model_ckpts/OOD_generalization/LBaug_embclip/exp_LBaug-ObjectNav-ProcTHOR-RGB-ClipResNet50GRU-DDPPO__stage_00__steps_000027371520.pt" if args.checkpoint is None else args.checkpoint
        base_tag        = "LBaug-ObjectNav-ProcTHOR-RGB-ClipResNet50GRU-DDPPO"

    # Read experiment config
    with open(exp_config_path, 'r') as f:
        exp_conf_data = f.readlines()
    
    # ProcTHOR OOD scenes test dataset
    dataset_dir_test = r"datasets/ProcTHOR/Test"

    # TARGET OBJECTS
    targets = [
                [ 'Television',
                'Sofa',
                'Newspaper'], 
                ['Fridge',
                'Kettle',
                'Apple'],  
                ['AlarmClock',
                'Bed', 
                'Dresser'] 
                ]


    target_rooms = ["Livingroom", "Kitchen", "Bedroom"]
    experiments = {0 : ["ALL"]*3,
                1 : target_rooms,
                2 : target_rooms,
                3 : ["ALL"]*3}

    episodes_per_targ_obj = ['24', '12', '3', '3']

    for n_wall_color_changes in range(4):
        for i, targ_objs in enumerate(targets):
            for targ_obj in targ_objs:
                # Find lines in experiment config file
                test_obj_line_ind =     [exp_conf_data.index(line) for line in exp_conf_data if re.match("\s*TEST_OBJECT_TYPES" ,line)][0]
                scene_dir_line_ind =    [exp_conf_data.index(line) for line in exp_conf_data if re.match("\s*HOUSES_DATASET_DIR_TEST" ,line)][0]
                test_samples_line_ind = [exp_conf_data.index(line) for line in exp_conf_data if re.match("\s*TEST_SAMPLES_IN_SCENE" ,line)][0]

                # Set new experiment params
                scene_dir = os.path.join(dataset_dir_test,f"{n_wall_color_changes}_{experiments[n_wall_color_changes][i]}")

                exp_conf_data[test_obj_line_ind]        = re.sub('sorted\(\[\"[A-Za-z]+\"',
                                                                 f'sorted([\"{targ_obj}\"', 
                                                                 exp_conf_data[test_obj_line_ind])
                exp_conf_data[scene_dir_line_ind]       = re.sub(dataset_dir_test + '/[0-9]_[A-Za-z]+', 
                                                                 scene_dir, 
                                                                 exp_conf_data[scene_dir_line_ind])
                exp_conf_data[test_samples_line_ind]    = re.sub('[0-9]+', 
                                                                 episodes_per_targ_obj[n_wall_color_changes], 
                                                                 exp_conf_data[test_samples_line_ind])

                # Write to experiment config file
                with open(exp_config_path, 'w') as f:
                    f.writelines(exp_conf_data)

                set_tag(exp_config_path, f'{n_wall_color_changes}_changes-{base_tag}')

                # Run experiment
                cmd = f"PYTHONPATH=. python allenact/main.py {os.path.basename(exp_config_path)} -o {args.test_output_dir}/{args.experiment} -b {args.experiment_base} -c {ckpt_path} --eval"
                                
                os.system(cmd) 

    return

def viz(args) -> None:
    exp_config_path = f'{args.experiment_base}/viz_objectnav_procthor_rgb_clipresnet50gru_ddppo.py'
    
    with open(exp_config_path, 'r') as f:
        exp_conf_data = f.readlines()

    class_line_ind = [exp_conf_data.index(line) for line in exp_conf_data if re.match("\s*class" ,line)][0]

    if args.experiment == 'closed_world_embclip':
        exp_conf_data[class_line_ind] = re.sub('\(.+\)','(CW_EmbCLIP_ObjectNavProcThorPPOExperimentConfig)', exp_conf_data[class_line_ind])
        ckpt_path       = r"pretrained_model_ckpts/OOD_generalization/closed_world_embclip/exp_ObjectNav-ProcTHOR-RGB-ClipResNet50GRU-DDPPO__stage_00__steps_000019906560.pt" if args.checkpoint is None else args.checkpoint
        new_tag = "CW-ObjectNav-ProcTHOR-RGB-ClipResNet50GRU-DDPPO"
    elif args.experiment == 'open_world_embclip':
        exp_conf_data[class_line_ind] = re.sub('\(.+\)','(OW_EmbCLIP_ObjectNavProcThorPPOExperimentConfig)', exp_conf_data[class_line_ind])
        ckpt_path       = r"pretrained_model_ckpts/OOD_generalization/open_world_embclip/exp_open_world-ObjectNav-ProcTHOR-RGB-ClipResNet50GRU-DDPPO__stage_00__steps_000029237760.pt" if args.checkpoint is None else args.checkpoint
        new_tag = "OW-ObjectNav-ProcTHOR-RGB-ClipResNet50GRU-DDPPO"
    elif args.experiment == 'LBaug_embclip':
        exp_conf_data[class_line_ind] = re.sub('\(.+\)','(LBaug_EmbCLIP_ObjectNavProcThorPPOExperimentConfig)', exp_conf_data[class_line_ind])
        ckpt_path       = r"pretrained_model_ckpts/OOD_generalization/LBaug_embclip/exp_LBaug-ObjectNav-ProcTHOR-RGB-ClipResNet50GRU-DDPPO__stage_00__steps_000027371520.pt" if args.checkpoint is None else args.checkpoint
        new_tag = "LBaug-ObjectNav-ProcTHOR-RGB-ClipResNet50GRU-DDPPO"

    with open(exp_config_path, 'w') as f:
        f.writelines(exp_conf_data)
    
    set_tag(exp_config_path, new_tag)

    cmd = f"PYTHONPATH=. python allenact/main.py {os.path.basename(exp_config_path)} -o {args.viz_output_dir} -b {args.experiment_base} -c {ckpt_path} --eval"

    os.system(cmd)
    return

def main():
    parser = get_argument_parser()
    args = parser.parse_args()

    if args.eval:
        test(args)
    elif args.viz:
        viz(args)
    else:
        train(args)

    return

if __name__ == "__main__":
    main()
