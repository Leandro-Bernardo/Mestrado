from argparse import Namespace
import argparse, inspect, os
import torch
from typing import Dict, Final, List, Optional, Type

from chemical_analysis import alkalinity, chloride, phosphate, sulfate, iron3, iron2, bisulfite2d, ph, sweep_dist2dist, Network, ContinuousUpNetwork
from chemical_analysis.sweep import ContinuousModel
from chemical_analysis.sweep_dist2dist import ContinuousAutoEncoder, AutoEncoderDataModule
from _const import AnalyteClasses, AnalyteName, AnalyteUpClasses, WandbMode, DevMode, Dist2DistArtificialValues

# Default values for development arguments
DEFAULT_DEV_MODE: Final[str] = DevMode.PROD
DEFAULT_LOG_FREQUENCE: Final[int] = 10

# Default values for Weights & Biases arguments.
DEFAULT_WANDB_ENTITY_NAME: Final[str] = "prograf-uff"
DEFAULT_WANDB_PROJECT_NAME_MASK: Final[str] = "chemical-analysis-dist2dist-{analyte:s}"
DEFAULT_WANDB_SWEEP_ID: Final[Optional[str]] = "i0ievxnh"
DEFAULT_WANDB_SWEEP_NAME: Final[Optional[str]] = "Encoder Vgg11 Cyclic" if DEFAULT_WANDB_SWEEP_ID is None else None
DEFAULT_WANDB_MODE: Final[str] = WandbMode.ONLINE # After offline training is necessary make the upload using 'wandb sync --include-offline .\wandb\offline-*'


# Default values for network model arguments.
REGRESSOR_NETWORK_CHOICES: Dict[str, AnalyteClasses] = {
    **{name: AnalyteClasses(analyte=AnalyteName.ALKALINITY, expected_range=(500.0, 2500.0), network_class=obj, sample_dataset_class=alkalinity.AlkalinitySampleDataset, processed_sample_dataset_class=alkalinity.ProcessedAlkalinitySampleDataset) for name, obj in inspect.getmembers(alkalinity) if inspect.isclass(obj) and issubclass(obj, alkalinity.AlkalinityNetwork) and obj != alkalinity.AlkalinityNetwork},
    **{name: AnalyteClasses(analyte=AnalyteName.CHLORIDE,   expected_range=(10000.0, 300000.0), network_class=obj, sample_dataset_class=chloride.ChlorideSampleDataset,     processed_sample_dataset_class=chloride.ProcessedChlorideSampleDataset)  for name, obj in inspect.getmembers(chloride) if inspect.isclass(obj) and issubclass(obj, (chloride.ChlorideNetwork, chloride.ChlorideIntervalNetwork)) and obj != chloride.ChlorideNetwork and obj != chloride.ChlorideIntervalNetwork},
    **{name: AnalyteClasses(analyte=AnalyteName.BISULFITE,  expected_range=(1.0, 20.0), network_class=obj, sample_dataset_class=bisulfite2d.Bisulfite2DSampleDataset,processed_sample_dataset_class=bisulfite2d.ProcessedBisulfite2DSampleDataset) for name, obj in inspect.getmembers(bisulfite2d) if inspect.isclass(obj) and issubclass(obj, bisulfite2d.Bisulfite2DNetwork) and obj != bisulfite2d.Bisulfite2DNetwork},
    **{name: AnalyteClasses(analyte=AnalyteName.PH, expected_range=(0.0, 14.0), network_class=obj, sample_dataset_class=ph.PhSampleDataset, processed_sample_dataset_class=ph.ProcessedPhSampleDataset) for name, obj in inspect.getmembers(ph) if inspect.isclass(obj) and issubclass(obj, ph.PhNetwork) and obj != ph.PhNetwork},
    **{name: AnalyteClasses(analyte=AnalyteName.SULFATE, expected_range=(0.0, 4000.0), network_class=obj, sample_dataset_class=sulfate.SulfateSampleDataset, processed_sample_dataset_class=sulfate.ProcessedSulfateSampleDataset) for name, obj in inspect.getmembers(sulfate) if inspect.isclass(obj) and issubclass(obj, sulfate.SulfateNetwork) and obj != sulfate.SulfateNetwork},
    **{name: AnalyteClasses(analyte=AnalyteName.PHOSPHATE, expected_range=(0.0, 50.0), network_class=obj, sample_dataset_class=phosphate.PhosphateSampleDataset, processed_sample_dataset_class=phosphate.ProcessedPhosphateSampleDataset)for name, obj in inspect.getmembers(phosphate) if inspect.isclass(obj) and issubclass(obj, phosphate.PhosphateNetwork) and obj != phosphate.PhosphateNetwork},
}

GENERATOR_NETWORK_CHOICES: Dict[str, AnalyteUpClasses] = {
    **{name: AnalyteUpClasses(analyte=AnalyteName.ALKALINITY, network_class=obj, sample_dataset_class=alkalinity.AlkalinitySampleDataset, processed_sample_dataset_class=alkalinity.ProcessedAlkalinitySampleDataset) for name, obj in inspect.getmembers(alkalinity) if inspect.isclass(obj) and issubclass(obj, alkalinity.AlkalinityUpNetwork) and obj != alkalinity.AlkalinityUpNetwork},
    **{name: AnalyteUpClasses(analyte=AnalyteName.CHLORIDE, network_class=obj, sample_dataset_class=chloride.ChlorideSampleDataset, processed_sample_dataset_class=chloride.ProcessedChlorideSampleDataset) for name, obj in inspect.getmembers(chloride) if inspect.isclass(obj) and issubclass(obj, chloride.ChlorideUpNetwork) and obj != chloride.ChlorideUpNetwork},
    **{name: AnalyteUpClasses(analyte=AnalyteName.BISULFITE, network_class=obj, sample_dataset_class=bisulfite2d.Bisulfite2DSampleDataset, processed_sample_dataset_class=bisulfite2d.ProcessedBisulfite2DSampleDataset) for name, obj in inspect.getmembers(bisulfite2d) if inspect.isclass(obj) and issubclass(obj, bisulfite2d.Bisulfite2DUpNetwork) and obj != bisulfite2d.Bisulfite2DUpNetwork},
    **{name: AnalyteUpClasses(analyte=AnalyteName.PH, network_class=obj, sample_dataset_class=ph.PhSampleDataset, processed_sample_dataset_class=ph.ProcessedPhSampleDataset) for name, obj in inspect.getmembers(ph) if inspect.isclass(obj) and issubclass(obj, ph.PhUpNetwork) and obj != ph.PhUpNetwork},
    **{name: AnalyteUpClasses(analyte=AnalyteName.SULFATE, network_class=obj, sample_dataset_class=sulfate.SulfateSampleDataset, processed_sample_dataset_class=sulfate.ProcessedSulfateSampleDataset) for name, obj in inspect.getmembers(sulfate) if inspect.isclass(obj) and issubclass(obj, sulfate.SulfateUpNetwork) and obj != sulfate.SulfateUpNetwork},
    **{name: AnalyteUpClasses(analyte=AnalyteName.PHOSPHATE, network_class=obj, sample_dataset_class=phosphate.PhosphateSampleDataset, processed_sample_dataset_class=phosphate.ProcessedPhosphateSampleDataset) for name, obj in inspect.getmembers(phosphate) if inspect.isclass(obj) and issubclass(obj, phosphate.PhosphateUpNetwork) and obj != phosphate.PhosphateUpNetwork},
}

DEFAULT_ORACLE_CLASS: Final[Type[Network]] = alkalinity.AlkalinityNetworkVgg11Style
DEFAULT_GENERATOR_CLASS: Final[Type[Network]] = alkalinity.AlkalinityNetworkUpVgg11Style

DEFAULT_ORACLE_CHECKPOINT: Final[Dict[str, str]] = {
    AnalyteName.ALKALINITY: alkalinity.NETWORK_CHECKPOINT,
    AnalyteName.CHLORIDE: chloride.NETWORK_CHECKPOINT,
    AnalyteName.PHOSPHATE: phosphate.NETWORK_CHECKPOINT,
    AnalyteName.SULFATE: sulfate.NETWORK_CHECKPOINT,
    AnalyteName.IRON2: iron2.NETWORK_CHECKPOINT,
    AnalyteName.IRON3: iron3.NETWORK_CHECKPOINT,
    AnalyteName.BISULFITE: bisulfite2d.NETWORK_CHECKPOINT,
    AnalyteName.PH: ph.NETWORK_CHECKPOINT,
}

DEFAULT_GENERATOR_CHECKPOINT: Final[Dict[str, str]] = {
    AnalyteName.ALKALINITY: alkalinity.UPNETWORK_CHECKPOINT,
    AnalyteName.CHLORIDE: chloride.UPNETWORK_CHECKPOINT,
    AnalyteName.PHOSPHATE: phosphate.UPNETWORK_CHECKPOINT,
    AnalyteName.SULFATE: sulfate.UPNETWORK_CHECKPOINT,
    AnalyteName.IRON2: iron2.UPNETWORK_CHECKPOINT,
    AnalyteName.IRON3: iron3.UPNETWORK_CHECKPOINT,
    AnalyteName.BISULFITE: bisulfite2d.UPNETWORK_CHECKPOINT,
    AnalyteName.PH: ph.UPNETWORK_CHECKPOINT,
}

# Default values for dataset arguments.
DATASET_ROOT: Final[str] = os.path.join(os.path.dirname(__file__), "..", "..")
DEFAULT_FIT_SAMPLES_BASE_DIRS: Final[Dict[str, List[str]]] = {
    AnalyteName.ALKALINITY: [
        os.path.join(DATASET_ROOT, "MABIDs-Dataset-Alkalinity", "CR11", "train"),
    ],
    AnalyteName.CHLORIDE: [
        os.path.join(DATASET_ROOT, "MABIDs-Dataset-Chloride", "train"),
    ],
    AnalyteName.PHOSPHATE: [],
    AnalyteName.SULFATE: [
        os.path.join(DATASET_ROOT, "MABIDs-Dataset-Sulfate", "train")
    ],
    AnalyteName.IRON2: [],
    AnalyteName.IRON3: [],
    AnalyteName.BISULFITE: [
        os.path.join(DATASET_ROOT, "MABIDs-Dataset-Bisulfite", "train")
    ],
    AnalyteName.PH: [
        os.path.join(DATASET_ROOT, "MABIDs-Dataset-pH", "train")
    ],
}
DEFAULT_TEST_SAMPLES_BASE_DIRS: Final[Dict[str, List[str]]] = {
    AnalyteName.ALKALINITY: [
        os.path.join(DATASET_ROOT, "MABIDs-Dataset-Alkalinity", "CR11", "test"),
    ],
    AnalyteName.CHLORIDE: [
        os.path.join(DATASET_ROOT, "MABIDs-Dataset-Chloride", "test"),
    ],
    AnalyteName.PHOSPHATE: [],
    AnalyteName.SULFATE: [
        os.path.join(DATASET_ROOT, "MABIDs-Dataset-Sulfate", "test")
    ],
    AnalyteName.IRON2: [],
    AnalyteName.IRON3: [],
    AnalyteName.BISULFITE: [
        os.path.join(DATASET_ROOT, "MABIDs-Dataset-Bisulfite", "test")
    ],
    AnalyteName.PH: [
        os.path.join(DATASET_ROOT, "MABIDs-Dataset-pH", "test")
    ],
}
DEFAULT_USE_EXPANDED_SET: Final[bool] = False
DEFAULT_NUM_AUGMENTED_SAMPLES: Final[int] = 0
DEFAULT_REDUCTION_LEVEL: Final[float] = 0.10
DEFAULT_VAL_PROPORTION: Final[float] = 0.30
# Default Dist2Dist parameters
DEFAULT_ARTIFICIAL_VALUES_DISTRIBUTION: Final[str] = Dist2DistArtificialValues.CYCLIC
DEFAULT_REAL_DATA_PERCENT: Final[float] = 1.0 # from 0 to 1
DEFAULT_ARTIFICIAL_DATA_PERCENT: Final[float] = 1.0 # from 0 to 1 EQUIDISTANCE works with intagers and RANDOM with floats
DEFAULT_INTERVALS_NUMBER: Final[Dict[str, int]] = {
    AnalyteName.ALKALINITY: 10,
    AnalyteName.CHLORIDE: 10,
    AnalyteName.PHOSPHATE: 10,
    AnalyteName.SULFATE: 10,
    AnalyteName.IRON2: 10,
    AnalyteName.IRON3: 10,
    AnalyteName.BISULFITE: 10,
    AnalyteName.PH: 10,
}
# Default values for general arguments.
DEFAULT_CHECKPOINT_DIR: Final[str] = "checkpoint"
DEFAULT_EARLY_STOPPING_PATIENCE: Final[int] = 20
DEFAULT_LEARNING_RATE_PATIENCE: Final[int] = 10
DEFAULT_GPUS: Final[int] = -1 if torch.cuda.is_available() else 0  # -1 stands for all GPUs available, and 0 stands for CPU (no GPU).
DEFAULT_SEED: Final[Optional[int]] = None  # None stands for random seed.


# The main method.
def main(args: Namespace) -> None:

    if args.dist2dist_oracle_checkpoint is None:
        raise Exception('Oracle checkpoint not informed')

    if issubclass(args.generator.network_class, ContinuousUpNetwork):
        generator_model_class = ContinuousAutoEncoder
    else:
        raise NotImplementedError
    torch.set_float32_matmul_precision("high")

    if args.execution_mode == DevMode.PROD or args.execution_mode == DevMode.DEBUG:
        if args.wandb_resume:
            # Resume an existing sweep.
            sweep_dist2dist.resume(
                # Dev arguments
                dev_exec_mode = args.execution_mode,
                dev_log_frequence = args.log_frequence,
                # Weights & Biases arguments.
                entity_name=args.wandb_entity,
                project_name=args.wandb_project,
                sweep_id=args.wandb_resume,
                sweep_mode=args.wandb_mode,
                # Network and run arguments.
                checkpoint_dir=args.checkpoint_dir,
                expected_range=args.oracle.expected_range,
                early_stopping_patience=args.early_stopping_patience,
                learning_rate_patience=args.learning_rate_patience,
                gpus=args.gpus,
                oracle_checkpoint=args.dist2dist_oracle_checkpoint,
                oracle_network_class=args.oracle.network_class,
                seed=args.seed,
                auto_encoder_model_class=generator_model_class,
                generator_network_class=args.generator.network_class,
                # Dataset arguments.
                dataset_root_dir=args.dataset_root_dir,
                fit_samples_base_dirs=args.fit_samples_base_dirs,
                num_augmented_samples=args.num_augmented_samples,
                processed_sample_dataset_class=args.oracle.processed_sample_dataset_class,
                reduction_level=args.reduction_level,
                sample_dataset_class=args.oracle.sample_dataset_class,
                test_samples_base_dirs=args.test_samples_base_dirs,
                use_expanded_set=args.use_expanded_set,
                val_proportion=args.val_proportion,
                use_artificial_data=False,
                # Data augmentation arguments
                values_distribution=args.dist2dist_artificial_values_distribution,
                real_data_percent=args.dist2dist_real_data_percent,
                artificial_data_percent=args.dist2dist_artificial_data_percent,
                intervals_number=args.dist2dist_intervals_number,
            )
        else:
            # Setup the sweep configuration.
            config = sweep_dist2dist.make_base_config(sweep_name=args.wandb_start, program=os.path.basename(__file__))
            config["metric"].update(generator_model_class.wandb_metric())
            config["parameters"].update(AutoEncoderDataModule.wandb_parameters())
            config["parameters"].update(generator_model_class.wandb_parameters())
            # Start a new sweep.
            sweep_dist2dist.start(
                # Dev arguments
                dev_exec_mode = args.execution_mode,
                dev_log_frequence = args.log_frequence,
                # Weights & Biases arguments.
                config=config,
                entity_name=args.wandb_entity,
                project_name=args.wandb_project,
                sweep_id=args.wandb_resume,
                sweep_mode=args.wandb_mode,
                # Network and run arguments.
                checkpoint_dir=args.checkpoint_dir,
                expected_range=args.oracle.expected_range,
                early_stopping_patience=args.early_stopping_patience,
                learning_rate_patience=args.learning_rate_patience,
                gpus=args.gpus,
                oracle_checkpoint=args.dist2dist_oracle_checkpoint,
                oracle_network_class=args.oracle.network_class,
                seed=args.seed,
                auto_encoder_model_class=generator_model_class,
                generator_network_class=args.generator.network_class,
                # Dataset arguments.
                dataset_root_dir=args.dataset_root_dir,
                fit_samples_base_dirs=args.fit_samples_base_dirs,
                num_augmented_samples=args.num_augmented_samples,
                processed_sample_dataset_class=args.oracle.processed_sample_dataset_class,
                reduction_level=args.reduction_level,
                sample_dataset_class=args.oracle.sample_dataset_class,
                test_samples_base_dirs=args.test_samples_base_dirs,
                use_expanded_set=args.use_expanded_set,
                val_proportion=args.val_proportion,
                use_artificial_data=False,
                # Data augmentation arguments
                values_distribution=args.dist2dist_artificial_values_distribution,
                real_data_percent=args.dist2dist_real_data_percent,
                artificial_data_percent=args.dist2dist_artificial_data_percent,
                intervals_number=args.dist2dist_intervals_number,
            )
    elif args.execution_mode == DevMode.GEN_DEBUG or args.execution_mode == DevMode.GEN_PROD:
        sweep_dist2dist._generate_artificial_pmfs(
            # Check points
            oracle_checkpoint=args.dist2dist_oracle_checkpoint,
            oracle_network_class=args.oracle.network_class,
            generator_checkpoint=args.dist2dist_generator_checkpoint,
            generator_network_class=args.generator.network_class,
            # Data augmentation arguments
            values_distribution=args.dist2dist_artificial_values_distribution,
            real_data_percent=args.dist2dist_real_data_percent,
            artificial_data_percent=args.dist2dist_artificial_data_percent,
            intervals_number=args.dist2dist_intervals_number,
            # Dataset arguments
            dataset_root_dir=args.dataset_root_dir,
            fit_samples_base_dirs=args.fit_samples_base_dirs,
            num_augmented_samples=args.num_augmented_samples,
            processed_sample_dataset_class=args.oracle.processed_sample_dataset_class,
            reduction_level=args.reduction_level,
            sample_dataset_class=args.oracle.sample_dataset_class,
            test_samples_base_dirs=args.test_samples_base_dirs,
            use_expanded_set=args.use_expanded_set,
            val_proportion=args.val_proportion,
            use_artificial_data=False,
        )
    else:
        raise(Exception(f"DevMode with value {args.execution_mode} is not maped!"))


# Call the main method.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Development arguments
    group = parser.add_argument_group("development arguments")
    group.add_argument("--execution_mode", type=str, default=DEFAULT_DEV_MODE, help="execution type of the scripts")
    group.add_argument("--log_frequence", type=str, default=DEFAULT_LOG_FREQUENCE, help="value that determines the frequence of full log")

    # Network model arguments.
    group = parser.add_argument_group("network model arguments")
    group.add_argument("--oracle", type=str, default=REGRESSOR_NETWORK_CHOICES[DEFAULT_ORACLE_CLASS.__name__], help="the name of the class of the network model to be loaded")
    group.add_argument("--generator", type=str, default=GENERATOR_NETWORK_CHOICES[DEFAULT_GENERATOR_CLASS.__name__], help="the name of the class of the generator network model to be trained")

    # Weights & Biases arguments.
    group = parser.add_argument_group("logger arguments")
    group.add_argument("--wandb_entity", metavar="NAME", type=str, default=DEFAULT_WANDB_ENTITY_NAME, help="the name of the entity in the Weights & Biases framework")
    group.add_argument("--wandb_project", metavar="NAME", type=str, default=DEFAULT_WANDB_PROJECT_NAME_MASK.format(analyte=REGRESSOR_NETWORK_CHOICES[DEFAULT_ORACLE_CLASS.__name__].analyte), help="the name of the project in the Weights & Biases framework")
    switch = group.add_mutually_exclusive_group()
    switch.add_argument("--wandb_start", metavar="NAME", type=str, default=DEFAULT_WANDB_SWEEP_NAME, help="the name of the sweep to be created in the Weights & Biases framework")
    switch.add_argument("--wandb_resume", metavar="ID", type=str, default=DEFAULT_WANDB_SWEEP_ID, help="the ID of the sweep to be resumed in the Weights & Biases framework")
    switch.add_argument("--wandb_mode", metavar="RUN_MODE", type=str, default=DEFAULT_WANDB_MODE, help="the run mode used to determine if it logs online, offline or syncs the logs")

    # Dataset arguments.
    group = parser.add_argument_group("dataset arguments")
    group = parser.add_argument_group("dataset arguments")
    group.add_argument("--fit_samples_base_dirs", metavar="PATHS", nargs="+", default=[], help="list of paths to folders with fit samples")
    group.add_argument("--test_samples_base_dirs", metavar="PATHS", nargs="+", default=[], help="list of paths to folders with fit samples")
    group.add_argument("--dataset_root_dir", metavar="PATH", type=str, help="path to the root dir where the dataset will be creates")
    switch = group.add_mutually_exclusive_group()
    switch.add_argument("--use_expanded_set", dest="use_expanded_set", action="store_true")
    switch.add_argument("--dont_use_expanded_set", dest="use_expanded_set", action="store_false")
    switch.set_defaults(use_expanded_set=DEFAULT_USE_EXPANDED_SET)
    group.add_argument("--num_augmented_samples", metavar="VALUE", type=int, default=DEFAULT_NUM_AUGMENTED_SAMPLES, help="number of augmented samples to be generated")
    group.add_argument("--reduction_level", metavar="VALUE", type=float, default=DEFAULT_REDUCTION_LEVEL, help="amount of less frequent calibrated a*b* samples that will be removed from the input distribution, VALUE in [0, 1]")
    group.add_argument("--val_proportion", metavar="VALUE", type=float, default=DEFAULT_VAL_PROPORTION, help="amount of dataset entries used as validation, VALUE in [0, 1]")

    # Dist2Dist arguments
    group.add_argument("--dist2dist_artificial_values_distribution", metavar="DIST2DIST", type=str, default=DEFAULT_ARTIFICIAL_VALUES_DISTRIBUTION, help="EQUIDISTANCE works with intagers and RANDOM with floats")
    group.add_argument("--dist2dist_real_data_percent", metavar="DIST2DIST", type=float, default=DEFAULT_REAL_DATA_PERCENT, help="percent of real data to be kept")
    group.add_argument("--dist2dist_artificial_data_percent", metavar="DIST2DIST", type=float, default=DEFAULT_ARTIFICIAL_DATA_PERCENT, help="percent of real data to be created")
    group.add_argument("--dist2dist_intervals_number", metavar="DIST2DIST", type=int, default=None, help="data value interval to be considered")
    group.add_argument("--dist2dist_oracle_checkpoint", metavar="DIST2DIST", type=str, default=None, help="path to the checkpoint of the model used as oracle")
    group.add_argument("--dist2dist_generator_checkpoint", metavar="DIST2DIST", type=str, default=None, help="path to the checkpoint of the model used as generator")

    # Set general arguments.
    group = parser.add_argument_group("general arguments")
    group.add_argument("--checkpoint_dir", metavar="PATH", type=str, default=DEFAULT_CHECKPOINT_DIR, help="the path to the model checkpoint folder")
    group.add_argument("--early_stopping_patience", metavar="EPOCHS", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE, help="early stopping epochs")
    group.add_argument("--learning_rate_patience", metavar="EPOCHS", type=int, default=DEFAULT_LEARNING_RATE_PATIENCE, help="early stopping epochs")
    group.add_argument("--gpus", metavar="COUNT", type=int, default=DEFAULT_GPUS, help=f"the number of GPUs used to train (0-{torch.cuda.device_count()}), or -1 to all")
    group.add_argument("--seed", metavar="VALUE", type=int, default=DEFAULT_SEED, help="the seed for generating random numbers while splitting the dataset and performing data augmentation")

    # Parse arguments.
    args = parser.parse_args()
    if args.wandb_project is None:
        args.wandb_project = DEFAULT_WANDB_PROJECT_NAME_MASK.format(analyte=args.oracle.analyte)
    if len(args.fit_samples_base_dirs) == 0:
        args.fit_samples_base_dirs = DEFAULT_FIT_SAMPLES_BASE_DIRS[args.oracle.analyte]
    if len(args.test_samples_base_dirs) == 0:
        args.test_samples_base_dirs = DEFAULT_TEST_SAMPLES_BASE_DIRS[args.oracle.analyte]
    if args.dataset_root_dir is None:
        args.dataset_root_dir = os.path.join(os.path.dirname(__file__), "dataset", args.oracle.analyte)
    if args.dist2dist_oracle_checkpoint is None:
        args.dist2dist_oracle_checkpoint = DEFAULT_ORACLE_CHECKPOINT[args.oracle.analyte]
    if args.dist2dist_generator_checkpoint is None:
        args.dist2dist_generator_checkpoint = DEFAULT_GENERATOR_CHECKPOINT[args.generator.analyte]
    if args.dist2dist_intervals_number is None:
        args.dist2dist_intervals_number = DEFAULT_INTERVALS_NUMBER[args.generator.analyte]
    # Call the main procedure.
    main(args)
