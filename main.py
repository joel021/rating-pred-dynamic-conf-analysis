import argparse
import glob
import json
import os
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
                              num_negatives=setup.num_negatives,
                              folds=setup.folds,
                              hyperparameters=setup.hyperparameters
                              )

        if setup_model_results_exists(environ.instance_dir) and not setup.reevaluate:
            print(f"All results already obtained for fold {fold}. Skip.")
            continue
    
        # Load model and dataloaders
        model, fit_dl, val_dl = environ.get_model_dataloaders(True)
        
        # Check if the model checkpoint exists
        model_exists = environ.model_uri and os.path.isfile(environ.model_uri)
        is_knn = 'knn' in setup.model_name.lower()
        
        # Fit model if requested (fit_mode == 0) and (not trained yet or reevaluation is forced)
        if setup.fit_mode == 0 and (setup.reevaluate or not model_exists or is_knn):
            print(f"Fitting model for fold {fold}...")
            model = setup_fit(setup, model, fit_dl, val_dl, environ, device)
            export_setup(environ, setup.to_dict())
        else:
            print(f"Model already trained/loaded for fold {fold}. Skipping fitting.")

        # Always run evaluation if we didn't skip the fold entirely
        print(f"Running evaluation for fold {fold}...")
        eval_df = export_elementwise_error(model, environ, device, fold)
        eval_metrics = evaluate(eval_df, environ)
        export_metrics(environ, {"eval": eval_metrics})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--setups", type=str, default="setups.json",
                        help="Path to predefined setups JSON file")
    parser.add_argument("--setup_name", type=str, default=None,
                        help="Specific setup name to run from the setups JSON")
    parser.add_argument("--k_folds", type=int, default=None,
                        help="Override number of folds")
    parser.add_argument("--fit_mode", type=int, default=None,
                        help="Override fit mode")
    parser.add_argument("--setup_instance", type=str, default=None,
                        help="Path to a setup instance directory to re-execute")

    args = parser.parse_args()

    if args.setup_instance:
        setup_files = glob.glob(os.path.join(args.setup_instance, "setup-[0-9]*.json"))
        if not setup_files:
            raise FileNotFoundError(f"No setup file found in {args.setup_instance}")
        setup_dict = read_json(setup_files[0])
        if args.fit_mode is not None:
            setup_dict['fit_mode'] = args.fit_mode
        setup = Setup(**setup_dict)
        main(setup)
    else:
        setups = read_json(args.setups)
        if args.setup_name:
            if args.setup_name not in setups:
                raise ValueError(f"Setup {args.setup_name} not found in {args.setups}")
            keys = [args.setup_name]
        else:
            keys = list(setups.keys())
            
        for key in keys:
            print(f"Running setup: {key}")
            setup_dict = setups[key]
            if args.k_folds is not None:
                setup_dict['folds'] = args.k_folds
            if args.fit_mode is not None:
                setup_dict['fit_mode'] = args.fit_mode
            setup = Setup(**setup_dict)
            main(setup)