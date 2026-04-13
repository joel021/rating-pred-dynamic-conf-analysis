import argparse
import glob
import json

import torch

from recsysconfident.environment import Environment
from recsysconfident.ml.eval.inference_error_analysis import export_elementwise_error
from recsysconfident.ml.eval.ranking_evaluation import evaluate
from recsysconfident.setup import Setup
from recsysconfident.utils.files import export_metrics, export_setup, read_json, \
    setup_and_model_exists, setup_model_results_exists
from recsysconfident.setup_manager import setup_fit


def main(setup: Setup):
    """
    shuffle_train_split: whether shuffle the train split or use sorted by timestamp
    """
    print(setup.to_dict())
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
    for fold in range(setup.folds-1):

        environ = Environment(model_name=setup.model_name,
                        database_name=setup.database_name,
                        split_position=fold,
                        batch_size=setup.batch_size,
                        min_inter_per_user=setup.min_inter_per_user
                        )

        if setup_model_results_exists(environ.instance_dir) and not setup.reevaluate:
            print("All results already obtained. Skip.")
            continue
    
        if setup.fit_mode == 0 and not setup_and_model_exists(environ.instance_dir):
            model, fit_dl, val_dl = environ.get_model_dataloaders(True)
            model = setup_fit(setup, model, fit_dl, val_dl, environ, device, fold)

            export_setup(environ, setup.to_dict())
            eval_df = export_elementwise_error(model, environ, device, fold)
            eval_metrics = evaluate(eval_df, environ)
            
            export_metrics(environ, {"eval": eval_metrics})


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--setups", type=str, default="setups.json",
                        help="Path to predefined setups JSON file")

    args = parser.parse_args()
    setups = read_json(args.setups)
    
    for key in setups.keys():
        
        print(setups[key])

        setup_dict = setups[key]
        setup = Setup(**setup_dict)
        main(setup)