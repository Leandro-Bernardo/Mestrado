from argparse import Namespace
from chemical_analysis import alkalinity, chloride, phosphate, sulfate, iron_oxid, iron3, iron2, bisulfite2d, ph, sweep, ContinuousNetwork, IntervalNetwork, Network
from chemical_analysis.sweep import DataModule, ContinuousModel, IntervalModel, ProcessedSampleDataset, SampleDataset
from typing import Dict, Final, List, NamedTuple, Optional, Tuple, Type
import argparse, inspect, os
import torch
from _const import WandbMode, AnalyteName


class AnalyteClasses(NamedTuple):
    analyte: str
    expected_range: Tuple[float, float]
    network_class: Type[Network]
    sample_dataset_class: Type[SampleDataset]
    processed_sample_dataset_class: Type[ProcessedSampleDataset]


# Default values for Weights & Biases arguments.
DEFAULT_WANDB_ENTITY_NAME: Final[str] = "uff-and-prograf"
DEFAULT_WANDB_PROJECT_NAME_MASK: Final[str] = "Chloride-reinjecao"
DEFAULT_WANDB_SWEEP_ID: Final[Optional[str]] = None
DEFAULT_WANDB_SWEEP_NAME: Final[Optional[str]] = "new sweep" if DEFAULT_WANDB_SWEEP_ID is None else None
DEFAULT_WANDB_MODE: Final[str] = WandbMode.ONLINE # After offline training is necessary make the upload using 'wandb sync --include-offline .\wandb\offline-*'

# Default values for network model arguments.
NETWORK_CHOICES: Dict[str, AnalyteClasses] = {
    **{name: AnalyteClasses(analyte=AnalyteName.ALKALINITY, expected_range=(500.0, 2500.0),     network_class=obj, sample_dataset_class=alkalinity.AlkalinitySampleDataset, processed_sample_dataset_class=alkalinity.ProcessedAlkalinitySampleDataset) for name, obj in inspect.getmembers(alkalinity) if inspect.isclass(obj) and issubclass(obj, alkalinity.AlkalinityNetwork) and obj != alkalinity.AlkalinityNetwork},
    **{name: AnalyteClasses(analyte=AnalyteName.CHLORIDE,   expected_range=(10000.0, 300000.0), network_class=obj, sample_dataset_class=chloride.ChlorideSampleDataset,     processed_sample_dataset_class=chloride.ProcessedChlorideSampleDataset)  for name, obj in inspect.getmembers(chloride) if inspect.isclass(obj) and issubclass(obj, (chloride.ChlorideNetwork, chloride.ChlorideIntervalNetwork)) and obj != chloride.ChlorideNetwork and obj != chloride.ChlorideIntervalNetwork},
    **{name: AnalyteClasses(analyte=AnalyteName.PHOSPHATE,  expected_range=(0.0, 50.0),         network_class=obj, sample_dataset_class=phosphate.PhosphateSampleDataset,   processed_sample_dataset_class=phosphate.ProcessedPhosphateSampleDataset)  for name, obj in inspect.getmembers(phosphate) if inspect.isclass(obj) and issubclass(obj, phosphate.PhosphateNetwork) and obj != phosphate.PhosphateNetwork},
    **{name: AnalyteClasses(analyte=AnalyteName.SULFATE,    expected_range=(0.0, 4000.0),       network_class=obj, sample_dataset_class=sulfate.SulfateSampleDataset,       processed_sample_dataset_class=sulfate.ProcessedSulfateSampleDataset)  for name, obj in inspect.getmembers(sulfate) if inspect.isclass(obj) and issubclass(obj, sulfate.SulfateNetwork) and obj != sulfate.SulfateNetwork},
    **{name: AnalyteClasses(analyte=AnalyteName.IRON_OXID,  expected_range=(0.0, 10.0),         network_class=obj, sample_dataset_class=iron_oxid.IronOxidSampleDataset,    processed_sample_dataset_class=iron_oxid.ProcessedIronOxidSampleDataset)  for name, obj in inspect.getmembers(iron_oxid) if inspect.isclass(obj) and issubclass(obj, iron_oxid.IronOxidNetwork) and obj != iron_oxid.IronOxidNetwork},
    **{name: AnalyteClasses(analyte=AnalyteName.IRON3,      expected_range=(0.0, 20.0),         network_class=obj, sample_dataset_class=iron3.Iron3SampleDataset,           processed_sample_dataset_class=iron3.ProcessedIron3SampleDataset)  for name, obj in inspect.getmembers(iron3) if inspect.isclass(obj) and issubclass(obj, iron3.Iron3Network) and obj != iron3.Iron3Network},
    **{name: AnalyteClasses(analyte=AnalyteName.IRON2,      expected_range=(0.0, 20.0),         network_class=obj, sample_dataset_class=iron2.Iron2SampleDataset,           processed_sample_dataset_class=iron2.ProcessedIron2SampleDataset)  for name, obj in inspect.getmembers(iron2) if inspect.isclass(obj) and issubclass(obj, iron2.Iron2Network) and obj != iron2.Iron2Network},
    **{name: AnalyteClasses(analyte=AnalyteName.BISULFITE,  expected_range=(1.0, 20.0),         network_class=obj, sample_dataset_class=bisulfite2d.Bisulfite2DSampleDataset,processed_sample_dataset_class=bisulfite2d.ProcessedBisulfite2DSampleDataset)  for name, obj in inspect.getmembers(bisulfite2d) if inspect.isclass(obj) and issubclass(obj, bisulfite2d.Bisulfite2DNetwork) and obj != bisulfite2d.Bisulfite2DNetwork},
    **{name: AnalyteClasses(analyte=AnalyteName.PH,         expected_range=(5.0, 9.0),          network_class=obj, sample_dataset_class=ph.PhSampleDataset,                 processed_sample_dataset_class=ph.ProcessedPhSampleDataset)  for name, obj in inspect.getmembers(ph) if inspect.isclass(obj) and issubclass(obj, ph.PhNetwork) and obj != ph.PhNetwork},
}

DEFAULT_NETWORK_CLASS: Final[Type[Network]] = chloride.ChlorideNetworkSqueezeNetStyle

# Default values for Dist2Dist usage
DEFAULT_DIST2DIST_USAGE = False

# Default values for dataset arguments.
DATASET_ROOT: Final[str] = os.path.join(os.path.dirname(__file__), "..", "splited_samples")

DEFAULT_TRAIN_FIT_SAMPLES_BASE_DIRS: Final[Dict[str, List[str]]] = {
    AnalyteName.ALKALINITY: [
      os.path.join(DATASET_ROOT, "MABIDs-Dataset-Alkalinity", "CR11", "train"),
    ],
    AnalyteName.CHLORIDE: [
      os.path.join(DATASET_ROOT, "Chloride", "train_samples"),
    ],
    # AnalyteName.PHOSPHATE: [
    #      os.path.join(DATASET_ROOT, "MABIDs-Dataset-Phosphate", "train")
    # ],
    # AnalyteName.SULFATE: [
    #   os.path.join(DATASET_ROOT, "MABIDs-Dataset-Sulfate", "train"),
    # ],
    # AnalyteName.IRON_OXID: [
    #    os.path.join(DATASET_ROOT, "MABIDs-Dataset-IronOxid", "train")
    # ],
    # AnalyteName.IRON2: [ # TODO remover após testes
    #     os.path.join(DATASET_ROOT, "MABIDs-Dataset-Iron2")
    # ],
    # AnalyteName.IRON3: [ # TODO remover após testes
    #    os.path.join(DATASET_ROOT, "MABIDs-Dataset-Iron3", "train")
    # ],
    # AnalyteName.BISULFITE: [
    #     os.path.join(DATASET_ROOT, "MABIDs-Dataset-Bisulfite", "train(sem amostras do CENPS)")
    # ],
    # AnalyteName.PH: [
    #     os.path.join(DATASET_ROOT, "MABIDs-Dataset-pH", "train")
    # ],
}
DEFAULT_VAL_FIT_SAMPLES_BASE_DIRS: Final[Dict[str, List[str]]] = {
    AnalyteName.ALKALINITY: [
      os.path.join(DATASET_ROOT, "MABIDs-Dataset-Alkalinity", "CR11", "train"),
    ],
    AnalyteName.CHLORIDE: [
      os.path.join(DATASET_ROOT, "Chloride","val_samples"),
    ],
    # AnalyteName.PHOSPHATE: [
    #      os.path.join(DATASET_ROOT, "MABIDs-Dataset-Phosphate", "train")
    # ],
    # AnalyteName.SULFATE: [
    #   os.path.join(DATASET_ROOT, "MABIDs-Dataset-Sulfate", "train"),
    # ],
    # AnalyteName.IRON_OXID: [
    #    os.path.join(DATASET_ROOT, "MABIDs-Dataset-IronOxid", "train")
    # ],
    # AnalyteName.IRON2: [ # TODO remover após testes
    #     os.path.join(DATASET_ROOT, "MABIDs-Dataset-Iron2")
    # ],
    # AnalyteName.IRON3: [ # TODO remover após testes
    #    os.path.join(DATASET_ROOT, "MABIDs-Dataset-Iron3", "train")
    # ],
    # AnalyteName.BISULFITE: [
    #     os.path.join(DATASET_ROOT, "MABIDs-Dataset-Bisulfite", "train(sem amostras do CENPS)")
    # ],
    # AnalyteName.PH: [
    #     os.path.join(DATASET_ROOT, "MABIDs-Dataset-pH", "train")
    # ],
}
DEFAULT_TEST_SAMPLES_BASE_DIRS: Final[Dict[str, List[str]]] = {
    AnalyteName.ALKALINITY: [
      os.path.join(DATASET_ROOT, "MABIDs-Dataset-Alkalinity", "CR11", "test"),
    ],
    AnalyteName.CHLORIDE: [
        os.path.join(DATASET_ROOT, "Chloride","test_samples"),
    ],
    # AnalyteName.PHOSPHATE: [
    #     os.path.join(DATASET_ROOT, "MABIDs-Dataset-Phosphate", "test"),
    # ],
    # AnalyteName.SULFATE: [
    #     os.path.join(DATASET_ROOT, "MABIDs-Dataset-Sulfate", "test")
    # ],
    # AnalyteName.IRON_OXID: [
    #     os.path.join(DATASET_ROOT, "MABIDs-Dataset-IronOxid", "test")
    # ],
    # AnalyteName.IRON2: [ # TODO remover após testes
    #     os.path.join(DATASET_ROOT, "MABIDs-Dataset-Iron2", "test")
    # ],
    # AnalyteName.IRON3: [ # TODO remover após testes
    #     os.path.join(DATASET_ROOT, "MABIDs-Dataset-Iron3", "test")
    # ],
    # AnalyteName.BISULFITE: [
    #     os.path.join(DATASET_ROOT, "MABIDs-Dataset-Bisulfite", "test")
    # ],
    # AnalyteName.PH: [
    #     os.path.join(DATASET_ROOT, "MABIDs-Dataset-pH", "test")
    # ],
}
DEFAULT_USE_EXPANDED_SET: Final[bool] = False
DEFAULT_NUM_AUGMENTED_SAMPLES: Final[int] = 0
DEFAULT_REDUCTION_LEVEL: Final[Dict[str, float]] = {
    AnalyteName.ALKALINITY: 0.05,
    AnalyteName.CHLORIDE: 0.10,
    AnalyteName.PHOSPHATE: 0.10,
    AnalyteName.SULFATE: 0.10,
    AnalyteName.IRON_OXID: 0.10,
    AnalyteName.IRON2: 0.10,
    AnalyteName.IRON3: 0.10,
    AnalyteName.BISULFITE: 0.10,
    AnalyteName.PH: 0.10,
}
DEFAULT_VAL_PROPORTION: Final[float] = 0.30

# Default values for general arguments.
DEFAULT_CHECKPOINT_DIR: Final[str] = "checkpoint"
DEFAULT_EARLY_STOPPING_PATIENCE: Final[int] = 10
DEFAULT_LEARNING_RATE_PATIENCE: Final[int] = 5
DEFAULT_GPUS: Final[int] = -1 if torch.cuda.is_available() else 0  # -1 stands for all GPUs available, and 0 stands for CPU (no GPU).
DEFAULT_SEED: Final[Optional[int]] = None  # None stands for random seed.

# The main method.
def main(args: Namespace) -> None:
    if issubclass(args.net.network_class, ContinuousNetwork):
        model_class = ContinuousModel
    elif issubclass(args.net.network_class, IntervalNetwork):
        model_class = IntervalModel
    else:
        raise NotImplementedError
    torch.set_float32_matmul_precision("high")

    if args.wandb_resume:
        # Resume an existing sweep.
        sweep.resume(
            # Weights & Biases arguments.
            entity_name=args.wandb_entity,
            project_name=args.wandb_project,
            sweep_id=args.wandb_resume,
            wandb_mode=args.wandb_mode,
            # Network and run arguments.
            checkpoint_dir=args.checkpoint_dir,
            early_stopping_patience=args.early_stopping_patience,
            expected_range=args.net.expected_range,
            learning_rate_patience=args.learning_rate_patience,
            gpus=args.gpus,
            model_class=model_class,
            network_class=args.net.network_class,
            seed=args.seed,
            # Dataset arguments.
            dataset_root_dir=args.dataset_root_dir,
            fit_samples_base_dirs=args.fit_samples_base_dirs,
            num_augmented_samples=args.num_augmented_samples,
            processed_sample_dataset_class=args.net.processed_sample_dataset_class,
            reduction_level=args.reduction_level,
            sample_dataset_class=args.net.sample_dataset_class,
            test_samples_base_dirs=args.test_samples_base_dirs,
            use_expanded_set=args.use_expanded_set,
            val_proportion=args.val_proportion,
            use_artificial_data=args.allow_dist2dist_pmfs,
        )
    else:
        # Setup the sweep configuration.
        config = sweep.make_base_config(sweep_name=args.wandb_start, program=os.path.basename(__file__))
        config["metric"].update(model_class.wandb_metric())
        config["parameters"].update(DataModule.wandb_parameters())
        config["parameters"].update(model_class.wandb_parameters())
        # Start a new sweep.
        sweep.start(
            # Weights & Biases arguments.
            config=config,
            entity_name=args.wandb_entity,
            project_name=args.wandb_project,
            sweep_id=args.wandb_resume,
            wandb_mode=args.wandb_mode,
            # Network and run arguments.
            checkpoint_dir=args.checkpoint_dir,
            expected_range=args.net.expected_range,
            early_stopping_patience=args.early_stopping_patience,
            learning_rate_patience=args.learning_rate_patience,
            gpus=args.gpus,
            model_class=model_class,
            network_class=args.net.network_class,
            seed=args.seed,
            # Dataset arguments.
            dataset_root_dir=args.dataset_root_dir,
            #fit_samples_base_dirs=args.fit_samples_base_dirs,
            fit_train_samples_base_dirs=args.fit_train_samples_base_dirs,
            fit_val_samples_base_dirs=args.fit_val_samples_base_dirs,
            num_augmented_samples=args.num_augmented_samples,
            processed_sample_dataset_class=args.net.processed_sample_dataset_class,
            reduction_level=args.reduction_level,
            sample_dataset_class=args.net.sample_dataset_class,
            test_samples_base_dirs=args.test_samples_base_dirs,
            use_expanded_set=args.use_expanded_set,
            val_proportion=args.val_proportion,
            use_artificial_data=args.allow_dist2dist_pmfs,
        )


# Call the main method.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Weights & Biases arguments.
    group = parser.add_argument_group("logger arguments")
    group.add_argument("--wandb_entity", metavar="NAME", type=str, default=DEFAULT_WANDB_ENTITY_NAME, help="the name of the entity in the Weights & Biases framework")
    group.add_argument("--wandb_project", metavar="NAME", type=str, default=None, help="the name of the project in the Weights & Biases framework")
    switch = group.add_mutually_exclusive_group()
    switch.add_argument("--wandb_start", metavar="NAME", type=str, default=DEFAULT_WANDB_SWEEP_NAME, help="the name of the sweep to be created in the Weights & Biases framework")
    switch.add_argument("--wandb_resume", metavar="ID", type=str, default=DEFAULT_WANDB_SWEEP_ID, help="the ID of the sweep to be resumed in the Weights & Biases framework")
    switch.add_argument("--wandb_mode", metavar="RUN_MODE", type=str, default=DEFAULT_WANDB_MODE, help="the run mode used to determine if it logs online, offline or syncs the logs")
    # Network model arguments.
    group = parser.add_argument_group("network model arguments")
    group.add_argument("--net", type=str, choices=sorted(NETWORK_CHOICES.keys()), default=DEFAULT_NETWORK_CLASS.__name__, help="the name of the class of the network model to be trained")
    # Dataset arguments.
    group = parser.add_argument_group("dataset arguments")
    #group.add_argument("--fit_samples_base_dirs", metavar="PATHS", nargs="+", default=[], help="list of paths to folders with fit samples")
    group.add_argument("--fit_train_samples_base_dirs", metavar="PATHS", nargs="+", default=[], help="list of paths to folders with fit samples")
    group.add_argument("--fit_val_samples_base_dirs", metavar="PATHS", nargs="+", default=[], help="list of paths to folders with fit samples")
    group.add_argument("--test_samples_base_dirs", metavar="PATHS", nargs="+", default=[], help="list of paths to folders with fit samples")
    group.add_argument("--dataset_root_dir", metavar="PATH", type=str, help="path to the root dir where the dataset will be creates")
    switch = group.add_mutually_exclusive_group()
    switch.add_argument("--use_expanded_set", dest="use_expanded_set", action="store_true")
    switch.add_argument("--dont_use_expanded_set", dest="use_expanded_set", action="store_false")
    switch.set_defaults(use_expanded_set=DEFAULT_USE_EXPANDED_SET)
    group.add_argument("--num_augmented_samples", metavar="VALUE", type=int, default=DEFAULT_NUM_AUGMENTED_SAMPLES, help="number of augmented samples to be generated")
    group.add_argument("--reduction_level", metavar="VALUE", type=float, choices=sorted(DEFAULT_REDUCTION_LEVEL.keys()), default=None, help="amount of less frequent calibrated a*b* samples that will be removed from the input distribution, VALUE in [0, 1]")
    group.add_argument("--val_proportion", metavar="VALUE", type=float, default=DEFAULT_VAL_PROPORTION, help="amount of dataset entries used as validation, VALUE in [0, 1]")
    # Set general arguments.
    group = parser.add_argument_group("general arguments")
    group.add_argument("--checkpoint_dir", metavar="PATH", type=str, default=DEFAULT_CHECKPOINT_DIR, help="the path to the model checkpoint folder")
    group.add_argument("--early_stopping_patience", metavar="EPOCHS", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE, help="early stopping epochs")
    group.add_argument("--learning_rate_patience", metavar="EPOCHS", type=int, default=DEFAULT_LEARNING_RATE_PATIENCE, help="early stopping epochs")
    group.add_argument("--gpus", metavar="COUNT", type=int, default=DEFAULT_GPUS, help=f"the number of GPUs used to train (0-{torch.cuda.device_count()}), or -1 to all")
    group.add_argument("--seed", metavar="VALUE", type=int, default=DEFAULT_SEED, help="the seed for generating random numbers while splitting the dataset and performing data augmentation")
    # Set Dist2Dist arguments.
    group = parser.add_argument_group("dist2dist arguments")
    group.add_argument("--allow_dist2dist_pmfs", metavar="VALUE", type=bool, default=DEFAULT_DIST2DIST_USAGE, help="the flag to allow artificially generated pmfs")
    # Parse arguments.
    args = parser.parse_args()
    args.net = NETWORK_CHOICES[args.net]
    if args.wandb_project is None:
        args.wandb_project = DEFAULT_WANDB_PROJECT_NAME_MASK.format(analyte=args.net.analyte)
    if len(args.fit_train_samples_base_dirs) == 0:
        args.fit_train_samples_base_dirs = DEFAULT_TRAIN_FIT_SAMPLES_BASE_DIRS[args.net.analyte]
    if len(args.fit_val_samples_base_dirs) == 0:
        args.fit_val_samples_base_dirs = DEFAULT_VAL_FIT_SAMPLES_BASE_DIRS[args.net.analyte]
    if len(args.test_samples_base_dirs) == 0:
        args.test_samples_base_dirs = DEFAULT_TEST_SAMPLES_BASE_DIRS[args.net.analyte]
    if args.dataset_root_dir is None:
        args.dataset_root_dir = os.path.join(os.path.dirname(__file__), "dataset", args.net.analyte)
    if args.reduction_level is None:
        args.reduction_level = DEFAULT_REDUCTION_LEVEL[args.net.analyte]
    # Call the main procedure.
    main(args)
