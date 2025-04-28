from argparse import ArgumentParser, Namespace
from chemical_analysis import alkalinity, chloride, phosphate, sulfate, bisulfite2d, iron2,iron3,iron_oxid, redox, ph, suspended
from chemical_analysis import WHITEBALANCE_STATS, EstimationFunction, ExpandedSampleDataset, Network, ProcessedSampleDataset, estimate_confidence_in_whitebalance, lab_to_bgr, lab_to_rgb
from chemical_analysis.typing import AuxiliarySolution, Sample
from _const import AnalyteName
from datetime import date, datetime
from enum import IntEnum, auto, unique
from io import BytesIO
from tqdm import tqdm
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union
from xlsxwriter import Workbook
from xlsxwriter.chart import Chart
from xlsxwriter.format import Format
import cv2
import numpy as np
import math, os, shutil, sys, tempfile
import scipy.stats
import torch
import xlsxwriter.utility

import matplotlib
matplotlib.use('agg')
from matplotlib import colors
import matplotlib.pyplot as plt


# Default values for network model arguments.

ANALYTE_CHOICES: Final[List[str]] = ["alkalinity", "bisulfite", "chloride", "emulsion", "experimental", "iron2", "iron3", "iron_oxid", "ph", "phosphate", "redox", "sulfate", "suspended"]
DEFAULT_ANALYTE: Final[str] = AnalyteName.CHLORIDE
DEFAULT_DEVICE_NAME: Final[str] = "cuda" if torch.cuda.is_available() else "cpu"


# Default values for input arguments.
DATASET_ROOT: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#DATASET_ROOT = "G:\\Dataset"
DEFAULT_SAMPLES_BASE_DIRS: Final[Dict[str, List[str]]] = {
    AnalyteName.ALKALINITY: [
      os.path.join(DATASET_ROOT, "MABIDs-Dataset-Alkalinity", "CR11", "train"),
    ],
    AnalyteName.CHLORIDE: [
      os.path.join(DATASET_ROOT, "MABIDs-Dataset-Chloride", "train"),
    ],
    AnalyteName.PHOSPHATE: [
         os.path.join(DATASET_ROOT, "MABIDs-Dataset-Phosphate", "train")
    ],
    AnalyteName.SULFATE: [
      os.path.join(DATASET_ROOT, "MABIDs-Dataset-Sulfate", "train"),
    ],
    "iron3": [
                    #os.path.join(DATASET_ROOT, "MABIDs-Dataset-Iron3"),
                    #os.path.join(DATASET_ROOT, "Iron3"),
                    #os.path.join(DATASET_ROOT, "Iron3", "Validacao", "Iron3ExperimentalSamples"),
    ],
    AnalyteName.IRON3: [ # TODO remover após testes
       os.path.join(DATASET_ROOT, "Iron3", "ferro_total")
    ],
    AnalyteName.IRON_OXID: [
       os.path.join(DATASET_ROOT, "Iron_oxid", "IronOxidTrainingSamples", 'Uso Interno', 'Uso Interno', 'this')
    ],
    AnalyteName.BISULFITE: [
        os.path.join(DATASET_ROOT, "MABIDs-Dataset-Bisulfite", "train(sem amostras do CENPS)")
    ],
    AnalyteName.CHLORIDE: [
        os.path.join(DATASET_ROOT, "Splited_samples", "Chloride", "test_samples")
    ],


}


# Default values for report arguments.
DEFAULT_CLEAR_TEMP_FILES: Final[bool] = True
DEFAULT_COMPUTE_OFFLINE_ESTIMATION: Final[bool] = True
DEFAULT_SKIP_BLANK_SAMPLES: Final[bool] = True
DEFAULT_SKIP_INCOMPLETE_SAMPLES: Final[bool] = True
DEFAULT_SKIP_INFERENCE_SAMPLES: Final[bool] = True
DEFAULT_SKIP_TRAINING_SAMPLES: Final[bool] = False
DEFAULT_USE_EXPANDED_SET: Final[bool] = False
DEFAULT_WRITE_PDFS: Final[bool] = False


# Default checkpoints.
DEFAULT_CHECKPOINT: Final[Dict[str, str]] = {
    "alkalinity": alkalinity.NETWORK_CHECKPOINT,
    "chloride": chloride.NETWORK_CHECKPOINT,
    "experimental": alkalinity.NETWORK_CHECKPOINT,
    "phosphate": phosphate.NETWORK_CHECKPOINT,
    "sulfate": sulfate.NETWORK_CHECKPOINT,
    "bisulfite": bisulfite2d.NETWORK_CHECKPOINT,
    "iron2": iron2.NETWORK_CHECKPOINT,
    "iron3": iron3.NETWORK_CHECKPOINT,
    "iron_oxid": iron_oxid.NETWORK_CHECKPOINT,
    #"iron22d": iron22d.NETWORK_CHECKPOINT,
    #"iron32d": iron32d.NETWORK_CHECKPOINT,
    "redox": redox.NETWORK_CHECKPOINT,
    "ph": ph.NETWORK_CHECKPOINT,
    "suspended": suspended.NETWORK_CHECKPOINT,
}


# Default PCA statistics.
DEFAULT_PCA_STATS: Final[Dict[str, str]] = {
    "chloride": chloride.PCA_STATS,
    "phosphate": phosphate.PCA_STATS,

    "bisulfite": bisulfite2d.PCA_STATS,
    "iron2": iron2.PCA_STATS,
    "iron3": iron3.PCA_STATS,
    "iron_oxid": iron_oxid.PCA_STATS,
    #"iron22d": iron22d.PCA_STATS,
    #"iron32d": iron32d.PCA_STATS,
   # "redox": redox.PCA_STATS,F
   # "ph": ph.PCA_STATS,
    "suspended": suspended.PCA_STATS,
}


# Worksheet name.
SAMPLES_WORKSHEET_NAME: Final[str] = "Amostras"
WILCOXON_WORKSHEET_NAME: Final[str] = "Wilcoxon"


# Chart colors.
ESTIMATED_VALUE_COLOR: Final[str] = "#CC3300"
TRUE_VALUE_COLOR: Final[str] = "#0033CC"
TRUE_VALUE_BOUNDS_2_5_COLOR: Final[str] = "#178AFD"
TRUE_VALUE_BOUNDS_5_COLOR: Final[str] = "#63AFFD"
TRUE_VALUE_BOUNDS_10_COLOR: Final[str] = "#99CCFF"
TRUE_VALUE_BOUNDS_20_COLOR: Final[str] = "#CCECFF"


# Tags attached to worksheet columns.
@unique
class ColumnTag(IntEnum):
    ID = auto()
    TRUE_VALUE = auto()
    ESTIMATED_VALUE = auto()
    ESTIMATED_VALUE_ABSOLUTE_ERROR = auto()
    ESTIMATED_VALUE_RELATIVE_ERROR = auto()
    OFFLINE_ESTIMATED_VALUE = auto()
    OFFLINE_ESTIMATED_VALUE_ABSOLUTE_ERROR = auto()
    OFFLINE_ESTIMATED_VALUE_RELATIVE_ERROR = auto()
    ABSOLUTE_ERROR_BOUND_2_5 = auto()
    ABSOLUTE_ERROR_BOUND_5 = auto()
    ABSOLUTE_ERROR_BOUND_10 = auto()
    ABSOLUTE_ERROR_BOUND_20 = auto()
    RELATIVE_ERROR_BOUND_2_5 = auto()
    RELATIVE_ERROR_BOUND_5 = auto()
    RELATIVE_ERROR_BOUND_10 = auto()
    RELATIVE_ERROR_BOUND_20 = auto()
    RELATIVE_ERROR_LOWER_BOUND_2_5 = auto()
    RELATIVE_ERROR_LOWER_BOUND_5 = auto()
    RELATIVE_ERROR_LOWER_BOUND_10 = auto()
    RELATIVE_ERROR_LOWER_BOUND_20 = auto()
    RELATIVE_ERROR_UPPER_BOUND_2_5 = auto()
    RELATIVE_ERROR_UPPER_BOUND_5 = auto()
    RELATIVE_ERROR_UPPER_BOUND_10 = auto()
    RELATIVE_ERROR_UPPER_BOUND_20 = auto()
    RELATIVE_ERROR_STACKED_LOWER_BOUND_2_5 = auto()
    RELATIVE_ERROR_STACKED_LOWER_BOUND_5 = auto()
    RELATIVE_ERROR_STACKED_LOWER_BOUND_10 = auto()
    RELATIVE_ERROR_STACKED_LOWER_BOUND_20 = auto()
    RELATIVE_ERROR_STACKED_UPPER_BOUND_2_5 = auto()
    RELATIVE_ERROR_STACKED_UPPER_BOUND_5 = auto()
    RELATIVE_ERROR_STACKED_UPPER_BOUND_10 = auto()
    RELATIVE_ERROR_STACKED_UPPER_BOUND_20 = auto()
    SIMPLE = auto()


# Return name and address (letter) of the first occurence of the given column tag.
def column_properties(columns: List[Tuple[str, Any, Any, ColumnTag]], column_tag: ColumnTag) -> Tuple[str, str]:
    for index, (heading, _, _, tag) in enumerate(columns):
        if tag == column_tag:
            name = xlsxwriter.utility.xl_col_to_name(index)
            assert name is not None
            return heading, name
    raise RuntimeError("Column tag not found")


# Extract the analyte from the given image using the given mask.
def crop_analyte(imgs: Tuple[np.ndarray, ...], analyte_msk: np.ndarray) -> Tuple[np.ndarray, ...]:
    analyte_cols_msk = analyte_msk.any(axis=0)
    analyte_rows_msk = analyte_msk.any(axis=1)
    height, width = analyte_msk.shape
    min_x = analyte_cols_msk.argmax(axis=0)
    max_x = width - np.flip(analyte_cols_msk).argmax(axis=0) - 1
    min_y = analyte_rows_msk.argmax(axis=0)
    max_y = height - np.flip(analyte_rows_msk).argmax(axis=0) - 1
    result: List[np.ndarray] = list()
    for img in imgs:
        result.append(img[min_y:max_y+1, min_x:max_x+1, ...])
    return tuple(result)


# Put the distribution of the sample in the figure.
def draw_distribution(ax: Any, title: str, distribution: np.ndarray, cmap: str, vmin: float, vmax: float) -> None:
    ax.set_title(title)
    if distribution.ndim == 1:
        my_cmap = plt.get_cmap(cmap)
        ax.bar(np.arange(len(distribution)), distribution, width=1.0, color=my_cmap((distribution - vmin) / (vmax - vmin)))
        ax.set_ylim(vmin, vmax)
        ax.axis("on")
    elif distribution.ndim == 2:
        ax.imshow(distribution, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        raise NotImplementedError


# Put the image of the sample and its masks in the figure.
def draw_image_and_masks(
    ax: np.ndarray,
    title: str,
    *,
    name: str,
    solutions: Optional[List[AuxiliarySolution]],
    source_stock_name: Optional[str],
    standard_volume: float,
    used_volume: float,
    volume_unit: str,
    true_value: Optional[float],
    estimated_value: Optional[float],
    offline_estimated_value: Optional[float],
    value_unit: str,
    notes: str,
    analyst_name: str,
    datetime: Optional[datetime],
    rgb_img: np.ndarray,
    bright_msk: np.ndarray,
    grid_msk: np.ndarray,
    analyte_msk: np.ndarray,
    lab_white: np.ndarray,
    rgb_white: np.ndarray,
    mean_lab: np.ndarray,
    mean_rgb: np.ndarray,
) -> None:
    extra_msk = np.logical_xor(bright_msk, grid_msk)
    height, width, _ = rgb_img.shape
    # Make images.
    ax[0].set_title(f'Dados da Amostra\n{name}')
    ax[0].imshow(np.broadcast_to(mean_rgb, rgb_img.shape))
    ax[0].text(
        width // 2,
        height // 2,
        (f'Solução(ões) Auxiliar(es)\n{", ".join(map(lambda item: item["name"], solutions))}\n\n' if solutions is not None else "") \
        + (f'Solução Estoque\n{source_stock_name}\n\n' if source_stock_name is not None else "") \
        + (f'Valor Verdadeiro\n{true_value:1.2f} {value_unit}{" após redução para {:1.1f} {}".format(used_volume, volume_unit) if used_volume != standard_volume else ""}\n\n' if true_value is not None else "") \
        + (f'Estimativa do App\n{estimated_value:1.2f} {value_unit}{" após redução para {:1.1f} {}".format(used_volume, volume_unit) if used_volume != standard_volume else ""}\n\n' if estimated_value is not None else "") \
        + (f'Estimativa Offline\n{offline_estimated_value:1.2f} {value_unit}{" após redução para {:1.1f} {}".format(used_volume, volume_unit) if used_volume != standard_volume else ""}\n\n' if offline_estimated_value is not None else "") \
        + f'Cor Média do Analito\n' \
        + f'L*a*b* = ({mean_lab[0]:1.2f}, {mean_lab[1]:1.2f}, {mean_lab[2]:1.2f})\n' \
        + f'RGB = ({mean_rgb[0]}, {mean_rgb[1]}, {mean_rgb[2]})\n' \
        + f'Pixels Utilizados = {analyte_msk.sum()}\n\n' \
        + f'Cor Branca de Referência\n' \
        + f'L*a*b* = ({lab_white[0]:1.2f}, {lab_white[1]:1.2f}, {lab_white[2]:1.2f})\n' \
        + f'RGB = ({rgb_white[0]}, {rgb_white[1]}, {rgb_white[2]})',
        fontsize="x-small",
        ha="center",
        va="center",
    )
    ax[0].annotate(
        f'Analista: {analyst_name}' \
        + (f'\nData: {datetime.strftime("%d/%m/%Y %H:%M")}' if datetime is not None else "") \
        + (f'\nNotes: {notes}' if len(notes) != 0 else ""),
        (0, 0),
        (0, -5),
        xycoords="axes fraction",
        textcoords="offset points",
        fontsize="xx-small",
        ha="left",
        va="top",
    )
    ax[1].set_title(f'Imagem da\n{title}')
    ax[1].imshow(rgb_img)
    ax[2].set_title("Máscara da Grade")
    ax[2].imshow(np.broadcast_to(np.expand_dims((255 * grid_msk).astype(np.uint8), axis=2), rgb_img.shape))
    ax[3].set_title("Máscara do Analito")
    ax[3].imshow(np.broadcast_to(np.expand_dims((255 * analyte_msk).astype(np.uint8), axis=2), rgb_img.shape))
    ax[4].set_title("Máscara Extra\n(Grade e Flash)")
    ax[4].imshow(np.broadcast_to(np.expand_dims((255 * extra_msk).astype(np.uint8), axis=2), rgb_img.shape))


# Make the chart having estimated vs. true values.
def make_estimated_vs_true_values_chart(workbook: Workbook, columns: List[Tuple[str, Any, Any, ColumnTag]], num_samples: int, estimated_value_tag: ColumnTag) -> Chart:
    estimated_value_heading, estimated_value_name = column_properties(columns, estimated_value_tag)
    true_value_heading, true_value_name = column_properties(columns, ColumnTag.TRUE_VALUE)
    _, lower_bound_2_5_name = column_properties(columns, ColumnTag.RELATIVE_ERROR_LOWER_BOUND_2_5)
    _, upper_bound_2_5_name = column_properties(columns, ColumnTag.RELATIVE_ERROR_UPPER_BOUND_2_5)
    _, lower_bound_5_name = column_properties(columns, ColumnTag.RELATIVE_ERROR_LOWER_BOUND_5)
    _, upper_bound_5_name = column_properties(columns, ColumnTag.RELATIVE_ERROR_UPPER_BOUND_5)
    _, lower_bound_10_name = column_properties(columns, ColumnTag.RELATIVE_ERROR_LOWER_BOUND_10)
    _, upper_bound_10_name = column_properties(columns, ColumnTag.RELATIVE_ERROR_UPPER_BOUND_10)
    _, lower_bound_20_name = column_properties(columns, ColumnTag.RELATIVE_ERROR_LOWER_BOUND_20)
    _, upper_bound_20_name = column_properties(columns, ColumnTag.RELATIVE_ERROR_UPPER_BOUND_20)
    true_values = f'={SAMPLES_WORKSHEET_NAME}!${true_value_name}$2:${true_value_name}${num_samples+1}'
    # Make the scatter chart.
    chart = workbook.add_chart({"type": "scatter"})
    assert chart is not None
    chart.set_plotarea({"border": {"color": "black"}})
    chart.set_x_axis({"name": estimated_value_heading, "label_position": "low", "major_gridlines": {"visible": True}, 'num_font':  {'rotation': -45}})
    chart.set_y_axis({"name": true_value_heading, "label_position": "low", "major_gridlines": {"visible": True}})
    chart.add_series({
        "name": "Erro Relativo (20%)",
        "categories": true_values,
        "values": f'={SAMPLES_WORKSHEET_NAME}!${upper_bound_20_name}$2:${upper_bound_20_name}${num_samples+1}',
        "marker": {"type": "none"},
        "line": {"color": TRUE_VALUE_BOUNDS_20_COLOR, "dash_type": "solid", "width": 1.0},
    })
    chart.add_series({
        "name": "Erro Relativo (10%)",
        "categories": true_values,
        "values": f'={SAMPLES_WORKSHEET_NAME}!${upper_bound_10_name}$2:${upper_bound_10_name}${num_samples+1}',
        "marker": {"type": "none"},
        "line": {"color": TRUE_VALUE_BOUNDS_10_COLOR, "dash_type": "solid", "width": 1.0},
    })
    chart.add_series({
        "name": "Erro Relativo (5%)",
        "categories": true_values,
        "values": f'={SAMPLES_WORKSHEET_NAME}!${upper_bound_5_name}$2:${upper_bound_5_name}${num_samples+1}',
        "marker": {"type": "none"},
        "line": {"color": TRUE_VALUE_BOUNDS_5_COLOR, "dash_type": "solid", "width": 1.0},
    })
    chart.add_series({
        "name": "Erro Relativo (2.5%)",
        "categories": true_values,
        "values": f'={SAMPLES_WORKSHEET_NAME}!${upper_bound_2_5_name}$2:${upper_bound_2_5_name}${num_samples+1}',
        "marker": {"type": "none"},
        "line": {"color": TRUE_VALUE_BOUNDS_2_5_COLOR, "dash_type": "solid", "width": 1.0},
    })
    chart.add_series({
        "name": "Referência",
        "categories": true_values,
        "values": true_values,
        "marker": {"type": "none"},
        "line": {"color": TRUE_VALUE_COLOR, "dash_type": "solid", "width": 0.25},
    })

    chart.add_series({
        "categories": true_values,
        "values": f'={SAMPLES_WORKSHEET_NAME}!${lower_bound_2_5_name}$2:${lower_bound_2_5_name}${num_samples+1}',
        "marker": {"type": "none"},
        "line": {"color": TRUE_VALUE_BOUNDS_2_5_COLOR, "dash_type": "solid", "width": 1.0},
    })
    chart.add_series({
        "categories": true_values,
        "values": f'={SAMPLES_WORKSHEET_NAME}!${lower_bound_5_name}$2:${lower_bound_5_name}${num_samples+1}',
        "marker": {"type": "none"},
        "line": {"color": TRUE_VALUE_BOUNDS_5_COLOR, "dash_type": "solid", "width": 1.0},
    })
    chart.add_series({
        "categories": true_values,
        "values": f'={SAMPLES_WORKSHEET_NAME}!${lower_bound_10_name}$2:${lower_bound_10_name}${num_samples+1}',
        "marker": {"type": "none"},
        "line": {"color": TRUE_VALUE_BOUNDS_10_COLOR, "dash_type": "solid", "width": 1.0},
    })
    chart.add_series({
        "categories": true_values,
        "values": f'={SAMPLES_WORKSHEET_NAME}!${lower_bound_20_name}$2:${lower_bound_20_name}${num_samples+1}',
        "marker": {"type": "none"},
        "line": {"color": TRUE_VALUE_BOUNDS_20_COLOR, "dash_type": "solid", "width": 1.0},
    })
    chart.add_series({
        "name": "Amostra",
        "categories": f'={SAMPLES_WORKSHEET_NAME}!${estimated_value_name}$2:${estimated_value_name}${num_samples+1}',
        "values": true_values,
        "marker": {"type": "circle", "size": 2, "border": {"color": ESTIMATED_VALUE_COLOR}, "fill": {"color": ESTIMATED_VALUE_COLOR}},
        "trendline": {"type": "linear", "name": "Tendência", "line": {"color": ESTIMATED_VALUE_COLOR, "dash_type": "solid", "width": 0.25}},
    })
    chart.set_legend({'delete_series': [ 4, 5, 6, 7, 8]})
    return chart


# Make the chart having samples vs. errors.
def make_sample_vs_error_chart(workbook: Workbook, columns: List[Tuple[str, Any, Any, ColumnTag]], num_samples: int, error_value_tag: ColumnTag, error_bound_2_5_tag: ColumnTag, error_bound_5_tag: ColumnTag, error_bound_10_tag: ColumnTag, error_bound_20_tag: ColumnTag) -> Chart:
    id_heading, id_name = column_properties(columns, ColumnTag.ID)
    error_value_heading, error_value_name = column_properties(columns, error_value_tag)
    _, error_bound_2_5_name = column_properties(columns, error_bound_2_5_tag)
    _, error_bound_5_name = column_properties(columns, error_bound_5_tag)
    _, error_bound_10_name = column_properties(columns, error_bound_10_tag)
    _, error_bound_20_name = column_properties(columns, error_bound_20_tag)
    categories = f'={SAMPLES_WORKSHEET_NAME}!${id_name}$2:${id_name}${num_samples+1}'
    # Make the area chart.
    area_chart = workbook.add_chart({"type": "area"})
    assert area_chart is not None
    area_chart.set_plotarea({"border": {"color": "black"}})
    area_chart.set_x_axis({"name": id_heading, "label_position": "low"})
    area_chart.set_y_axis({"name": error_value_heading, "label_position": "low"})
    area_chart.add_series({
        "name": "Erro Relativo (20%)",
        "categories": categories,
        "values": f'={SAMPLES_WORKSHEET_NAME}!${error_bound_20_name}$2:${error_bound_20_name}${num_samples+1}',
        "fill": {"color": TRUE_VALUE_BOUNDS_20_COLOR},
        "border": {"none": True},
    })
    area_chart.add_series({
        "name": "Erro Relativo (10%)",
        "categories": categories,
        "values": f'={SAMPLES_WORKSHEET_NAME}!${error_bound_10_name}$2:${error_bound_10_name}${num_samples+1}',
        "fill": {"color": TRUE_VALUE_BOUNDS_10_COLOR},
        "border": {"none": True},
    })
    area_chart.add_series({
        "name": "Erro Relativo (5%)",
        "categories": categories,
        "values": f'={SAMPLES_WORKSHEET_NAME}!${error_bound_5_name}$2:${error_bound_5_name}${num_samples+1}',
        "fill": {"color": TRUE_VALUE_BOUNDS_5_COLOR},
        "border": {"none": True},
    })
    area_chart.add_series({
        "name": "Erro Relativo (2.5%)",
        "categories": categories,
        "values": f'={SAMPLES_WORKSHEET_NAME}!${error_bound_2_5_name}$2:${error_bound_2_5_name}${num_samples+1}',
        "fill": {"color": TRUE_VALUE_BOUNDS_2_5_COLOR},
        "border": {"none": True},
    })
    # Make the line chart.
    line_chart = workbook.add_chart({"type": "line"})
    assert line_chart is not None
    line_chart.add_series({
        "name": error_value_heading,
        "categories": categories,
        "values": f'={SAMPLES_WORKSHEET_NAME}!${error_value_name}$2:${error_value_name}${num_samples+1}',
        "marker": {"type": "circle", "size": 2, "border": {"color": ESTIMATED_VALUE_COLOR}, "fill": {"color": ESTIMATED_VALUE_COLOR}},
        "line": {"color": ESTIMATED_VALUE_COLOR, "dash_type": "solid", "width": 0.25},
    })
    # Combine charts.
    area_chart.combine(line_chart)
    return area_chart


# Make the chart having samples vs. estimated values.
def make_sample_vs_estimated_value_chart(workbook: Workbook, columns: List[Tuple[str, Any, Any, ColumnTag]], num_samples: int, estimated_value_tag: ColumnTag) -> Chart:
    id_heading, id_name = column_properties(columns, ColumnTag.ID)
    estimated_value_heading, estimated_value_name = column_properties(columns, estimated_value_tag)
    true_value_heading, true_value_name = column_properties(columns, ColumnTag.TRUE_VALUE)
    _, lower_bound_2_5_name = column_properties(columns, ColumnTag.RELATIVE_ERROR_STACKED_LOWER_BOUND_2_5)
    _, upper_bound_2_5_name = column_properties(columns, ColumnTag.RELATIVE_ERROR_STACKED_UPPER_BOUND_2_5)
    _, lower_bound_5_name = column_properties(columns, ColumnTag.RELATIVE_ERROR_STACKED_LOWER_BOUND_5)
    _, upper_bound_5_name = column_properties(columns, ColumnTag.RELATIVE_ERROR_STACKED_UPPER_BOUND_5)
    _, lower_bound_10_name = column_properties(columns, ColumnTag.RELATIVE_ERROR_STACKED_LOWER_BOUND_10)
    _, upper_bound_10_name = column_properties(columns, ColumnTag.RELATIVE_ERROR_STACKED_UPPER_BOUND_10)
    _, lower_bound_20_name = column_properties(columns, ColumnTag.RELATIVE_ERROR_STACKED_LOWER_BOUND_20)
    _, upper_bound_20_name = column_properties(columns, ColumnTag.RELATIVE_ERROR_STACKED_UPPER_BOUND_20)
    categories = f'={SAMPLES_WORKSHEET_NAME}!${id_name}$2:${id_name}${num_samples+1}'
    # Make the area chart.
    area_chart = workbook.add_chart({"type": "area", "subtype": "stacked"})
    assert area_chart is not None
    area_chart.set_plotarea({"border": {"color": "black"}})
    area_chart.set_x_axis({"name": id_heading, "label_position": "low"})
    area_chart.set_y_axis({"name": estimated_value_heading, "label_position": "low"})

    # this is the overlay
    area_chart.add_series({
        "categories": categories,
        "values": f'={SAMPLES_WORKSHEET_NAME}!${lower_bound_20_name}$2:${lower_bound_20_name}${num_samples+1}',
        "fill": {"none": True},
        "border": {"none": True},
    })
    area_chart.add_series({
        "categories": categories,
        "values": f'={SAMPLES_WORKSHEET_NAME}!${lower_bound_10_name}$2:${lower_bound_10_name}${num_samples+1}',
        "fill": {"color": TRUE_VALUE_BOUNDS_20_COLOR},
        "border": {"none": True},
    })

    area_chart.add_series({
        "categories": categories,
        "values": f'={SAMPLES_WORKSHEET_NAME}!${lower_bound_5_name}$2:${lower_bound_5_name}${num_samples+1}',
        "fill": {"color": TRUE_VALUE_BOUNDS_10_COLOR},
        "border": {"none": True},
    })
    area_chart.add_series({
        "categories": categories,
        "values": f'={SAMPLES_WORKSHEET_NAME}!${lower_bound_2_5_name}$2:${lower_bound_2_5_name}${num_samples+1}',
        "fill": {"color": TRUE_VALUE_BOUNDS_5_COLOR},
        "border": {"none": True},
    })
    # this is the graph itself
    area_chart.add_series({
        "name": "Erro Relativo (2.5%)",
        "categories": categories,
        "values": f'={SAMPLES_WORKSHEET_NAME}!${upper_bound_2_5_name}$2:${upper_bound_2_5_name}${num_samples+1}',
        "fill": {"color": TRUE_VALUE_BOUNDS_2_5_COLOR},
        "border": {"none": True},
    })
    area_chart.add_series({
        "name": "Erro Relativo (5%)",
        "categories": categories,
        "values": f'={SAMPLES_WORKSHEET_NAME}!${upper_bound_5_name}$2:${upper_bound_5_name}${num_samples+1}',
        "fill": {"color": TRUE_VALUE_BOUNDS_5_COLOR},
        "border": {"none": True},
    })
    area_chart.add_series({
        "name": "Erro Relativo (10%)",
        "categories": categories,
        "values": f'={SAMPLES_WORKSHEET_NAME}!${upper_bound_10_name}$2:${upper_bound_10_name}${num_samples+1}',
        "fill": {"color": TRUE_VALUE_BOUNDS_10_COLOR},
        "border": {"none": True},
    })
    area_chart.add_series({
        "name": "Erro Relativo (20%)",
        "categories": categories,
        "values": f'={SAMPLES_WORKSHEET_NAME}!${upper_bound_20_name}$2:${upper_bound_20_name}${num_samples+1}',
        "fill": {"color": TRUE_VALUE_BOUNDS_20_COLOR},
        "border": {"none": True},
    })
    area_chart.set_legend({'delete_series': [0, 1, 2, 3]})
    # Make the line chart.
    line_chart = workbook.add_chart({"type": "line"})
    assert line_chart is not None
    line_chart.set_plotarea({"border": {"color": "black"}})
    line_chart.add_series({
        "name": true_value_heading,
        "categories": categories,
        "values": f'={SAMPLES_WORKSHEET_NAME}!${true_value_name}$2:${true_value_name}${num_samples+1}',
        "marker": {"type": "none"},
        "line": {"color": TRUE_VALUE_COLOR, "dash_type": "solid", "width": 0.25},
    })
    line_chart.add_series({
        "name": estimated_value_heading,
        "categories": categories,
        "values": f'={SAMPLES_WORKSHEET_NAME}!${estimated_value_name}$2:${estimated_value_name}${num_samples+1}',
        "marker": {"type": "circle", "size": 2, "border": {"color": ESTIMATED_VALUE_COLOR}, "fill": {"color": ESTIMATED_VALUE_COLOR}},
        "line": {"color": ESTIMATED_VALUE_COLOR, "dash_type": "solid", "width": 0.25},
    })
    # Combine charts.
    area_chart.combine(line_chart)
    return area_chart


# Return the relative path of the PDF related to the given sample.
def pdf_relative_path(sample: Sample) -> str:
    datetime = sample["datetime"]
    basename, _ = os.path.splitext(os.path.basename(sample["fileName"]))
    return os.path.join(f'{datetime.year:4d}-{datetime.month:02d}-{datetime.day:02d}', f'{basename}.pdf')


# Compute the relative contribution of PMF entries to the estimated value.
def relative_contribution_at_pmfs(func: EstimationFunction, *, sample_pmf: np.ndarray, blank_pmf: np.ndarray, calibrated_pmf: np.ndarray, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Compute the Jacobian.
    jacobian_sample_pmf, jacobian_blank_pmf = torch.autograd.functional.jacobian(lambda arg1, arg2: func(sample_pmf=arg1, blank_pmf=arg2), (torch.as_tensor(sample_pmf, dtype=torch.float32, device=device), torch.as_tensor(blank_pmf, dtype=torch.float32, device=device)))
    jacobian_calibrated_pmf, = torch.autograd.functional.jacobian(lambda arg: func(calibrated_pmf=arg), (torch.as_tensor(calibrated_pmf, dtype=torch.float32, device=device).unsqueeze(0),))
    jacobian_calibrated_pmf = jacobian_calibrated_pmf.view(*calibrated_pmf.shape)
    # Compute the relative contribution of the input entries.
    relative_contribution_sample_pmf = torch.abs(jacobian_sample_pmf)
    relative_contribution_blank_pmf = torch.abs(jacobian_blank_pmf)
    total = relative_contribution_sample_pmf.sum() + relative_contribution_blank_pmf.sum()
    relative_contribution_sample_pmf /= total
    relative_contribution_blank_pmf /= total
    relative_contribution_calibrated_pmf = torch.abs(jacobian_calibrated_pmf)
    relative_contribution_calibrated_pmf /= relative_contribution_calibrated_pmf.sum()
    # Return the activation maps.
    return relative_contribution_sample_pmf.cpu().numpy(), relative_contribution_blank_pmf.cpu().numpy(), relative_contribution_calibrated_pmf.cpu().numpy()


# Compute the relative contribution of image pixels to the estimated value.
def relative_contribution_at_img(bgr_img: np.ndarray, relative_contribution_pmf: np.ndarray, img_to_pmf: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    img_coords, pmf_coords = img_to_pmf
    img_indices = tuple(img_coords.T)
    pmf_indices = pmf_coords if pmf_coords.ndim == 1 else tuple(pmf_coords.T)
    height, width, _ = bgr_img.shape
    # Create the mask of relative contribution for the image.
    count = np.zeros(relative_contribution_pmf.shape, dtype=np.int32)
    np.add.at(count, pmf_indices, 1)
    mask = np.zeros((height, width), dtype=np.float32)
    np.add.at(mask, img_indices, relative_contribution_pmf[pmf_indices])
    np.divide.at(mask, img_indices, count[pmf_indices])
    # Return the resulting mask.
    return mask


# Write the workbook.
def write_workbook(results_base_dir: str, dataset: ProcessedSampleDataset, net: Network, include_pdfs: bool) -> None:
    os.makedirs(results_base_dir, exist_ok=True)
    # Create the workbook.
    with Workbook(os.path.join(results_base_dir, f'Inventário - {date.today().strftime("%d.%m.%Y")}.xlsx'), options={"remove_timezone": True}) as workbook:
        # Set some usefun local function.
        def get(*field: Union[str, int]) -> Callable[[Sample], Any]:
            def _get(sample: Sample) -> Any:
                value: Any = sample
                for key in field:
                    value = value[key]
                return value if value is not None else ""
            return _get
        def get_basename(*field: str) -> Callable[[Sample], Any]:
            def _get_basename(sample: Sample) -> Any:
                value: Any = sample
                for key in field:
                    value = value[key]
                return os.path.basename(value) if value is not None else ""
            return _get_basename
        def get_image(*field: str) -> Callable[[Sample], Any]:
            def _get_image(sample: Sample) -> np.ndarray:
                value: Any = sample
                for key in field:
                    value = value[key]
                return cv2.imread(value, cv2.IMREAD_COLOR)
            return _get_image
        def get_logical(*field: str) -> Callable[[Sample], str]:
            def _get_logical(sample: Sample) -> str:
                value: Any = sample
                for key in field:
                    value = value[key]
                return "Sim" if value else "Não"
            return _get_logical
        def get_number(*field: Union[str, int]) -> Callable[[Sample], Any]:
            def _get(sample: Sample) -> Any:
                value: Any = sample
                for key in field:
                    value = value[key]
                return value if value is not None and not math.isnan(value) and math.isfinite(value) else ""
            return _get
        def hex_rgb(rgb: Tuple[float, float, float]) -> str:
            return f'#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}'
        def r_channel_fmt(*field: str) -> Callable[[Sample], Format]:
            def r_channel_fmt_func(sample: Sample) -> Format:
                rgb: Any = sample
                for key in field:
                    rgb = rgb[key]
                return workbook.add_format({"num_format": "0.00", "bg_color": hex_rgb((rgb[0], 0.0, 0.0))})
            return r_channel_fmt_func
        def g_channel_fmt(*field: str) -> Callable[[Sample], Format]:
            def g_channel_fmt_func(sample: Sample) -> Format:
                rgb: Any = sample
                for key in field:
                    rgb = rgb[key]
                return workbook.add_format({"num_format": "0.00", "bg_color": hex_rgb((0.0, rgb[1], 0.0))})
            return g_channel_fmt_func
        def b_channel_fmt(*field: str) -> Callable[[Sample], Format]:
            def b_channel_fmt_func(sample: Sample) -> Format:
                rgb: Any = sample
                for key in field:
                    rgb = rgb[key]
                return workbook.add_format({"num_format": "0.00", "bg_color": hex_rgb((0.0, 0.0, rgb[0]))})
            return b_channel_fmt_func
        def l_channel_fmt(*field: str) -> Callable[[Sample], Format]:
            def l_channel_fmt_func(sample: Sample) -> Format:
                lab: Any = sample
                for key in field:
                    lab = lab[key]
                return workbook.add_format({"num_format": "0.00", "bg_color": hex_rgb((lab[0], lab[0], lab[0]))})
            return l_channel_fmt_func
        def ab_channels_fmt(*field: str) -> Callable[[Sample], Format]:
            def ab_channel_fmt_func(sample: Sample) -> Format:
                lab: Any = sample
                for key in field:
                    lab = lab[key]
                bgr = lab_to_bgr(lab)
                bgr[0] = 128.0
                return workbook.add_format({"num_format": "0.00", "bg_color": hex_rgb((bgr[2], bgr[1], bgr[0]))})
            return ab_channel_fmt_func
        def mean_color_fmt(*field: str) -> Callable[[Sample], Format]:
            def mean_color_fmt_func(sample: Sample) -> Format:
                rgb: Any = sample
                for key in field:
                    rgb = rgb[key]
                return workbook.add_format({"bg_color": hex_rgb((rgb[0], rgb[1], rgb[2]))})
            return mean_color_fmt_func
        _datetime_fmt = workbook.add_format({"num_format": "dd/mm/yyyy hh:mm"})
        def datetime_fmt(_: Sample) -> Format:
            return _datetime_fmt
        _factor_fmt = workbook.add_format({"num_format": "0.000"})
        def factor_fmt(_: Sample) -> Format:
            return _factor_fmt
        def free_fmt(_: Sample) -> None:
            return None
        _percent_fmt = workbook.add_format({"num_format": "0.0%"})
        def percent_fmt(_: Sample) -> Format:
            return _percent_fmt
        _url_fmt = workbook.add_format({"align": "center_across", "font_color": "blue", "underline": 1})
        def url_fmt(_: Sample) -> Format:
            return _url_fmt
        _value_fmt: Dict[str, Format] = dict()
        def value_fmt(*field: str) -> Callable[[Sample], Format]:
            def value_fmt_func(sample: Sample) -> Format:
                unit: Any = sample
                for key in field:
                    unit = unit[key]
                fmt: Optional[Format] = _value_fmt.get(unit, None)
                if fmt is None:
                    fmt = workbook.add_format({"num_format": f'0.00 "{unit}"'})
                    _value_fmt[unit] = fmt
                return fmt
            return value_fmt_func
        _volume_fmt: Dict[str, Format] = dict()
        def volume_fmt(*field: str) -> Callable[[Sample], Format]:
            def volume_fmt_func(sample: Sample) -> Format:
                unit: Any = sample
                for key in field:
                    unit = unit[key]
                fmt = _volume_fmt.get(unit, None)
                if fmt is None:
                    fmt = workbook.add_format({"num_format": f'0.00 "{unit}"'})
                    _volume_fmt[unit] = fmt
                return fmt
            return volume_fmt_func
        _chart_fmt = workbook.add_format({"font_color": "#C0C0C0"})
        def chart_free_fmt(_: Sample) -> Format:
            return _chart_fmt
        _chart_percent_fmt = workbook.add_format({"num_format": "0.0%", "font_color": "#C0C0C0"})
        def chart_percent_fmt(_: Sample) -> Format:
            return _chart_percent_fmt
        def chart_value_fmt(*field: str) -> Callable[[Sample], Format]:
            def value_fmt_func(sample: Sample) -> Format:
                unit: Any = sample
                for key in field:
                    unit = unit[key]
                fmt: Optional[Format] = _value_fmt.get(f'{unit} [chart]', None)
                if fmt is None:
                    fmt = workbook.add_format({"num_format": f'0.00 "{unit}"', "font_color": "#C0C0C0"})
                    _value_fmt[f'{unit} [chart]'] = fmt
                return fmt
            return value_fmt_func
        # Add worksheet.
        sheet = workbook.add_worksheet(SAMPLES_WORKSHEET_NAME)
        sheet.set_default_row(hide_unused_rows=True)
        # Map sample fields to worksheet columns and formats.
        columns: List[Tuple[str, Callable[[Sample], Optional[Format]], Callable[[Sample], Any], ColumnTag]] = [
            ("Solução(ões) Auxiliar(es)",                          free_fmt,                              lambda sample: ", ".join(map(lambda item: item["name"], sample["auxiliarySolutions"])), ColumnTag.SIMPLE),
            ("Solução Estoque",                                    free_fmt,                              get("sourceStock", "name"),                                ColumnTag.SIMPLE),
            ("Valor na Solução Estoque",                           value_fmt("sourceStock", "valueUnit"), get_number("sourceStock", "value"),                        ColumnTag.SIMPLE),
            ("Diluição/Nome da Amostra",                           free_fmt,                              get("name"),                                               ColumnTag.SIMPLE),
            ("Valor na Diluição",                                  value_fmt("valueUnit"),                get_number("theoreticalValue"),                            ColumnTag.SIMPLE),
            ("Volume Padrão",                                      volume_fmt("volumeUnit"),              get_number("standardVolume"),                              ColumnTag.SIMPLE),
            ("Volume Utilizado",                                   volume_fmt("volumeUnit"),              get_number("usedVolume"),                                  ColumnTag.SIMPLE),
            ("Fator de Solução Estoque",                           factor_fmt,                            get_number("stockFactor"),                                 ColumnTag.SIMPLE),
            ("Valor Verdadeiro",                                   value_fmt("valueUnit"),                get_number("correctedTheoreticalValue"),                   ColumnTag.TRUE_VALUE),
            ("Estimativa do App",                                  value_fmt("valueUnit"),                get_number("estimatedValue"),                              ColumnTag.ESTIMATED_VALUE),
            ("Erro Absoluto",                                      value_fmt("valueUnit"),                get_number("extra", "estimatedValueAbsoluteError"),        ColumnTag.ESTIMATED_VALUE_ABSOLUTE_ERROR),
            ("Erro Relativo",                                      percent_fmt,                           get_number("extra", "estimatedValueRelativeError"),        ColumnTag.ESTIMATED_VALUE_RELATIVE_ERROR),
            ("Estimativa Offline",                                 value_fmt("valueUnit"),                get_number("extra", "offlineEstimatedValue"),              ColumnTag.OFFLINE_ESTIMATED_VALUE),
            ("Erro Absoluto",                                      value_fmt("valueUnit"),                get_number("extra", "offlineEstimatedValueAbsoluteError"), ColumnTag.OFFLINE_ESTIMATED_VALUE_ABSOLUTE_ERROR),
            ("Erro Relativo",                                      percent_fmt,                           get_number("extra", "offlineEstimatedValueRelativeError"), ColumnTag.OFFLINE_ESTIMATED_VALUE_RELATIVE_ERROR),
            ("Amostra de Zeragem",                                 free_fmt,                              get_logical("isBlankSample"),                              ColumnTag.SIMPLE),
            ("Amostra de Inferência",                              free_fmt,                              get_logical("isInferenceSample"),                          ColumnTag.SIMPLE),
            ("Amostra de Treino",                                  free_fmt,                              get_logical("isTrainingSample"),                           ColumnTag.SIMPLE),
            ("Solução de Zeragem Associada",                       free_fmt,                              get_logical("blankFileName"),                              ColumnTag.SIMPLE),
            ("Data de Captura",                                    datetime_fmt,                          get("datetime"),                                           ColumnTag.SIMPLE),
            ("Analista",                                           free_fmt,                              get("analystName"),                                        ColumnTag.SIMPLE),
            ("Modelo do Dispositivo",                              free_fmt,                              get("device", "model"),                                    ColumnTag.SIMPLE),
            ("Fabricante do Dispositivo",                          free_fmt,                              get("device", "manufacturer"),                             ColumnTag.SIMPLE),
            ("Versão do Android",                                  free_fmt,                              get("device", "androidVersion"),                           ColumnTag.SIMPLE),
            ("Versão do App",                                      free_fmt,                              get("app", "versionName"),                           ColumnTag.SIMPLE),
            ("Notas",                                              free_fmt,                              get("notes"),                                              ColumnTag.SIMPLE),
            ("Confiança no White Balance",                         percent_fmt,                           get_number("extra", "confidenceInWhiteBalance"),           ColumnTag.SIMPLE),
            ("R",                                                  r_channel_fmt("extra", "meanRGB"),     get_number("extra", "meanRGB", 0),                         ColumnTag.SIMPLE),
            ("G",                                                  g_channel_fmt("extra", "meanRGB"),     get_number("extra", "meanRGB", 1),                         ColumnTag.SIMPLE),
            ("B",                                                  b_channel_fmt("extra", "meanRGB"),     get_number("extra", "meanRGB", 2),                         ColumnTag.SIMPLE),
            ("L",                                                  l_channel_fmt("extra", "meanLab"),     get_number("extra", "meanLab", 0),                         ColumnTag.SIMPLE),
            ("a",                                                  ab_channels_fmt("extra", "meanLab"),   get_number("extra", "meanLab", 1),                         ColumnTag.SIMPLE),
            ("b",                                                  ab_channels_fmt("extra", "meanLab"),   get_number("extra", "meanLab", 2),                         ColumnTag.SIMPLE),
            ("Cor Média",                                          mean_color_fmt("extra", "meanRGB"),    lambda _: "",                                              ColumnTag.SIMPLE),
            ("Amostra",                                            free_fmt,                              get_image("extra", "croppedSampleFileName"),               ColumnTag.SIMPLE),
            *([("Arquivo PDF",                                     url_fmt,                               pdf_relative_path,                                         ColumnTag.SIMPLE),] if include_pdfs else []),
            ("Imagem da Amostra",                                  free_fmt,                              get_basename("fileName"),                                  ColumnTag.SIMPLE),
            ("Imagem da Amostra de Zeragem",                       free_fmt,                              get_basename("blankFileName"),                             ColumnTag.SIMPLE),
            # Special columns used by char
            ("Amostra",                                            chart_free_fmt,                        lambda sample: f'{sample["name"]} ({sample["datetime"].strftime("%d/%m/%Y %H:%M")})',                                  ColumnTag.ID),
            ("Limite do Erro Absoluto (20%)",                      chart_value_fmt("valueUnit"),          lambda sample: sample["correctedTheoreticalValue"] * 0.20 if sample["correctedTheoreticalValue"] is not None else "",  ColumnTag.ABSOLUTE_ERROR_BOUND_20),
            ("Limite do Erro Absoluto (10%)",                      chart_value_fmt("valueUnit"),          lambda sample: sample["correctedTheoreticalValue"] * 0.10 if sample["correctedTheoreticalValue"] is not None else "",  ColumnTag.ABSOLUTE_ERROR_BOUND_10),
            ("Limite do Erro Absoluto (5%)",                       chart_value_fmt("valueUnit"),          lambda sample: sample["correctedTheoreticalValue"] * 0.05 if sample["correctedTheoreticalValue"] is not None else "",  ColumnTag.ABSOLUTE_ERROR_BOUND_5),
            ("Limite do Erro Absoluto (2.5%)",                     chart_value_fmt("valueUnit"),          lambda sample: sample["correctedTheoreticalValue"] * 0.025 if sample["correctedTheoreticalValue"] is not None else "", ColumnTag.ABSOLUTE_ERROR_BOUND_2_5),
            ("Limite do Erro Relativo (20%)",                      chart_percent_fmt,                     lambda sample: 0.20 if sample["correctedTheoreticalValue"] is not None else "",                                        ColumnTag.RELATIVE_ERROR_BOUND_20),
            ("Limite do Erro Relativo (10%)",                      chart_percent_fmt,                     lambda sample: 0.10 if sample["correctedTheoreticalValue"] is not None else "",                                        ColumnTag.RELATIVE_ERROR_BOUND_10),
            ("Limite do Erro Relativo (5%)",                       chart_percent_fmt,                     lambda sample: 0.05 if sample["correctedTheoreticalValue"] is not None else "",                                        ColumnTag.RELATIVE_ERROR_BOUND_5),
            ("Limite do Erro Relativo (2.5%)",                     chart_percent_fmt,                     lambda sample: 0.025 if sample["correctedTheoreticalValue"] is not None else "",                                       ColumnTag.RELATIVE_ERROR_BOUND_2_5),
            ("Limite Superior do Erro Relativo (20%)",             chart_value_fmt("valueUnit"),          lambda sample: sample["correctedTheoreticalValue"] * 1.20 if sample["correctedTheoreticalValue"] is not None else "",  ColumnTag.RELATIVE_ERROR_UPPER_BOUND_20),
            ("Limite Superior do Erro Relativo (10%)",             chart_value_fmt("valueUnit"),          lambda sample: sample["correctedTheoreticalValue"] * 1.10 if sample["correctedTheoreticalValue"] is not None else "",  ColumnTag.RELATIVE_ERROR_UPPER_BOUND_10),
            ("Limite Superior do Erro Relativo (5%)",              chart_value_fmt("valueUnit"),          lambda sample: sample["correctedTheoreticalValue"] * 1.05 if sample["correctedTheoreticalValue"] is not None else "",  ColumnTag.RELATIVE_ERROR_UPPER_BOUND_5),
            ("Limite Superior do Erro Relativo (2.5%)",            chart_value_fmt("valueUnit"),          lambda sample: sample["correctedTheoreticalValue"] * 1.025 if sample["correctedTheoreticalValue"] is not None else "", ColumnTag.RELATIVE_ERROR_UPPER_BOUND_2_5),
            ("Limite Inferior do Erro Relativo (2.5%)",            chart_value_fmt("valueUnit"),          lambda sample: sample["correctedTheoreticalValue"] * 0.975 if sample["correctedTheoreticalValue"] is not None else "", ColumnTag.RELATIVE_ERROR_LOWER_BOUND_2_5),
            ("Limite Inferior do Erro Relativo (5%)",              chart_value_fmt("valueUnit"),          lambda sample: sample["correctedTheoreticalValue"] * 0.95 if sample["correctedTheoreticalValue"] is not None else "",  ColumnTag.RELATIVE_ERROR_LOWER_BOUND_5),
            ("Limite Inferior do Erro Relativo (10%)",             chart_value_fmt("valueUnit"),          lambda sample: sample["correctedTheoreticalValue"] * 0.90 if sample["correctedTheoreticalValue"] is not None else "",  ColumnTag.RELATIVE_ERROR_LOWER_BOUND_10),
            ("Limite Inferior do Erro Relativo (20%)",             chart_value_fmt("valueUnit"),          lambda sample: sample["correctedTheoreticalValue"] * 0.80 if sample["correctedTheoreticalValue"] is not None else "",  ColumnTag.RELATIVE_ERROR_LOWER_BOUND_20),
            ("Limite Superior do Erro Relativo (20%) - Empilhado", chart_value_fmt("valueUnit"),          lambda sample: sample["correctedTheoreticalValue"] * 0.10 if sample["correctedTheoreticalValue"] is not None else "", ColumnTag.RELATIVE_ERROR_STACKED_UPPER_BOUND_20),
            ("Limite Superior do Erro Relativo (10%) - Empilhado", chart_value_fmt("valueUnit"),          lambda sample: sample["correctedTheoreticalValue"] * 0.05 if sample["correctedTheoreticalValue"] is not None else "", ColumnTag.RELATIVE_ERROR_STACKED_UPPER_BOUND_10),
            ("Limite Superior do Erro Relativo (5%) - Empilhado", chart_value_fmt("valueUnit"),           lambda sample: sample["correctedTheoreticalValue"] * 0.025 if sample["correctedTheoreticalValue"] is not None else "", ColumnTag.RELATIVE_ERROR_STACKED_UPPER_BOUND_5),
            ("Limite Superior do Erro Relativo (2.5%) - Empilhado", chart_value_fmt("valueUnit"),         lambda sample: sample["correctedTheoreticalValue"] * 0.05 if sample["correctedTheoreticalValue"] is not None else "", ColumnTag.RELATIVE_ERROR_STACKED_UPPER_BOUND_2_5),
            ("Limite Inferior do Erro Relativo (2.5%) - Empilhado", chart_value_fmt("valueUnit"),         lambda sample: sample["correctedTheoreticalValue"] * 0.025 if sample["correctedTheoreticalValue"] is not None else "", ColumnTag.RELATIVE_ERROR_STACKED_LOWER_BOUND_2_5),
            ("Limite Inferior do Erro Relativo (5%) - Empilhado", chart_value_fmt("valueUnit"),           lambda sample: sample["correctedTheoreticalValue"] * 0.05 if sample["correctedTheoreticalValue"] is not None else "",  ColumnTag.RELATIVE_ERROR_STACKED_LOWER_BOUND_5),
            ("Limite Inferior do Erro Relativo (10%) - Empilhado", chart_value_fmt("valueUnit"),          lambda sample: sample["correctedTheoreticalValue"] * 0.10 if sample["correctedTheoreticalValue"] is not None else "",  ColumnTag.RELATIVE_ERROR_STACKED_LOWER_BOUND_10),
            ("Limite Inferior do Erro Relativo (20%) - Empilhado", chart_value_fmt("valueUnit"),          lambda sample: sample["correctedTheoreticalValue"] * 0.80 if sample["correctedTheoreticalValue"] is not None else "",  ColumnTag.RELATIVE_ERROR_STACKED_LOWER_BOUND_20),
        ]
        # Write header in the worksheet.
        title_fmt = workbook.add_format({"bg_color": "#BFBFBF", "bold": True})
        min_column_width = 7  # In characters.
        for col, (title, _, _, _) in enumerate(columns):
            sheet.write(0, col, title, title_fmt)
            sheet.set_column(col, col, max(len(title), min_column_width), None)
        row_height = 18  # Same as 24 in character units.
        character_width = 7.5  # Same as 1 in character's height units.
        magnification_factor = 8.0
        # Sort rows.
        sorted_samples: List[Sample] = [processed_sample.sample for processed_sample in tqdm(iter(dataset), "Sorting rows", total=len(dataset), leave=True)]
        sorted_samples.sort(key=lambda sample: \
            sample["correctedTheoreticalValue"] if sample["correctedTheoreticalValue"] is not None else \
            sample["estimatedValue"] if sample["estimatedValue"] is not None else \
            sample["extra"]["offlineEstimatedValue"] if sample["extra"] is not None and sample["extra"]["offlineEstimatedValue"] is not None else \
            float(-sys.maxsize)
        )
        # Write rows.
        true_values: List[float] = list()
        estimated_values: List[float] = list()
        num_below_2_5_percent, num_below_5_percent, num_below_10_percent, num_below_20_percent = 0, 0, 0, 0
        for row, sample in enumerate(tqdm(sorted_samples, "Writing worksheet", leave=True), start=1):
            assert sample["extra"] is not None
            true_value = sample["correctedTheoreticalValue"]
            estimated_value = sample["extra"]["offlineEstimatedValue"]
            relative_error = sample["extra"]["offlineEstimatedValueRelativeError"]
            if true_value is not None and estimated_value is not None and relative_error is not None:
                true_values.append(true_value)
                estimated_values.append(estimated_value)
                if relative_error <= 0.025:
                    num_below_2_5_percent += 1
                if relative_error <= 0.05:
                    num_below_5_percent += 1
                if relative_error <= 0.10:
                    num_below_10_percent += 1
                if relative_error <= 0.20:
                    num_below_20_percent += 1
            for col, (title, fmt_func, value_func, _) in enumerate(columns):
                value = value_func(sample)
                if isinstance(value, np.ndarray):
                    scale = magnification_factor * row_height / value.shape[0]
                    value = cv2.resize(value, (int(round(scale * value.shape[1])), int(round(scale * value.shape[0]))), interpolation=cv2.INTER_LANCZOS4)
                    sheet.insert_image(row, col, "", {"image_data": BytesIO(cv2.imencode(".png", value)[1].tobytes()), "x_scale": 1 / magnification_factor, "y_scale": 1 / magnification_factor, "x_offset": 0.5 * (max(len(title), min_column_width) * character_width - value.shape[1] / magnification_factor), "object_position": 1})
                elif fmt_func == url_fmt:
                    sheet.write_url(row, col, value_func(sample), fmt_func(sample), string="PDF")
                else:
                    sheet.write(row, col, value_func(sample), fmt_func(sample))
        # Enable filters.
        sheet.autofilter(0, 0, len(dataset) + 1, len(columns))
        # Use the Wilcoxon signed-rank test to test the null hypothesis that two related paired samples come from the same distribution.
        if len(true_values) != 0:
            alpha = 0.05
            _, pvalue = scipy.stats.wilcoxon(estimated_values, true_values)
            expected_range = net.expected_range
            bold = workbook.add_format({"bold": True})
            sheet = workbook.add_worksheet(WILCOXON_WORKSHEET_NAME)
            sheet.set_default_row(hide_unused_rows=True)
            sheet.write_rich_string("A1", "Utilizamos o modelo ", bold, net.version, ",")
            sheet.write("A2", f'nas estimativas offline. Por definição, a faixa de valores amostrais esperados vai de {expected_range[0]:1.2f} a {expected_range[1]:1.2f}.')
            sheet.write("A3", " ")
            sheet.write("A4", f'Utilizamos o teste de Wilcoxon para testar a hipótese nula (H0) de que pares')
            sheet.write("A5", f'de valores amostrais e estimados vêm da mesma distribuição.')
            sheet.write_rich_string("A6", f'Com valor-P = {pvalue:1.2f} e nível de significância alpha = {alpha:1.2f}, ', bold, "rejeitamos" if pvalue <= alpha else "não rejeitamos", " H0.")
            sheet.write("A7", " ")
            sheet.write_rich_string("A8",  bold, "Erro relativo na margem de 2.5%: ", f'{num_below_2_5_percent} de {len(true_values)} ({100.0 * (num_below_2_5_percent / len(true_values)):.1f}%)')
            sheet.write_rich_string("A9",  bold, "Erro relativo na margem de 5%: ",   f'{num_below_5_percent} de {len(true_values)} ({100.0 * (num_below_5_percent / len(true_values)):.1f}%)')
            sheet.write_rich_string("A10", bold, "Erro relativo na margem de 10%: ",  f'{num_below_10_percent} de {len(true_values)} ({100.0 * (num_below_10_percent / len(true_values)):.1f}%)')
            sheet.write_rich_string("A11", bold, "Erro relativo na margem de 20%: ",  f'{num_below_20_percent} de {len(true_values)} ({100.0 * (num_below_20_percent / len(true_values)):.1f}%)')
        # Make charts.
        def add_chartsheet(name: str, chart: Chart) -> None:
            chart.set_title({"name": name})
            sheet = workbook.add_worksheet(name)
            sheet.insert_chart(0, 0, chart, {'x_offset': 25, 'y_offset': 10})
        add_chartsheet("Tendência (App)", make_estimated_vs_true_values_chart(workbook, columns, len(dataset), ColumnTag.ESTIMATED_VALUE))
        add_chartsheet("Estimativa (App)", make_sample_vs_estimated_value_chart(workbook, columns, len(dataset), ColumnTag.ESTIMATED_VALUE))
        add_chartsheet("Erro Absoluto (App)", make_sample_vs_error_chart(workbook, columns, len(dataset), ColumnTag.ESTIMATED_VALUE_ABSOLUTE_ERROR, ColumnTag.ABSOLUTE_ERROR_BOUND_2_5, ColumnTag.ABSOLUTE_ERROR_BOUND_5, ColumnTag.ABSOLUTE_ERROR_BOUND_10, ColumnTag.ABSOLUTE_ERROR_BOUND_20))
        add_chartsheet("Erro Relativo (App)", make_sample_vs_error_chart(workbook, columns, len(dataset), ColumnTag.ESTIMATED_VALUE_RELATIVE_ERROR, ColumnTag.RELATIVE_ERROR_BOUND_2_5, ColumnTag.RELATIVE_ERROR_BOUND_5, ColumnTag.RELATIVE_ERROR_BOUND_10, ColumnTag.RELATIVE_ERROR_BOUND_20))
        add_chartsheet("Tendência (Offline)", make_estimated_vs_true_values_chart(workbook, columns, len(dataset), ColumnTag.OFFLINE_ESTIMATED_VALUE))
        add_chartsheet("Estimativa (Offline)", make_sample_vs_estimated_value_chart(workbook, columns, len(dataset), ColumnTag.OFFLINE_ESTIMATED_VALUE))
        add_chartsheet("Erro Absoluto (Offline)", make_sample_vs_error_chart(workbook, columns, len(dataset), ColumnTag.OFFLINE_ESTIMATED_VALUE_ABSOLUTE_ERROR, ColumnTag.ABSOLUTE_ERROR_BOUND_2_5, ColumnTag.ABSOLUTE_ERROR_BOUND_5, ColumnTag.ABSOLUTE_ERROR_BOUND_10, ColumnTag.ABSOLUTE_ERROR_BOUND_20))
        add_chartsheet("Erro Relativo (Offline)", make_sample_vs_error_chart(workbook, columns, len(dataset), ColumnTag.OFFLINE_ESTIMATED_VALUE_RELATIVE_ERROR, ColumnTag.RELATIVE_ERROR_BOUND_2_5, ColumnTag.RELATIVE_ERROR_BOUND_5, ColumnTag.RELATIVE_ERROR_BOUND_10, ColumnTag.RELATIVE_ERROR_BOUND_20))


# The main function.
@torch.no_grad()
def main(args: Namespace) -> None:
    torch.set_float32_matmul_precision("high")
    if args.clear_temp_files:
        root_dir = tempfile.mkdtemp(prefix=f'.{args.analyte}-report-data-', dir=os.path.dirname(__file__))
    else:
        root_dir = os.path.join(os.path.dirname(__file__), f'.{args.analyte}-report-persistent-data')
        os.makedirs(root_dir, exist_ok=True)
    try:
        # Get specialized classes.
        if args.analyte in ("alkalinity", "experimental"):
            estimation_func_class = alkalinity.AlkalinityEstimationFunction
            sample_dataset_class = alkalinity.AlkalinitySampleDataset
            processed_sample_dataset_class = alkalinity.ProcessedAlkalinitySampleDataset
        elif args.analyte == "chloride":
            estimation_func_class = chloride.ChlorideEstimationFunction
            sample_dataset_class = chloride.ChlorideSampleDataset
            processed_sample_dataset_class = chloride.ProcessedChlorideSampleDataset
        elif args.analyte == "phosphate":
            estimation_func_class = phosphate.PhosphateEstimationFunction
            sample_dataset_class = phosphate.PhosphateSampleDataset
            processed_sample_dataset_class = phosphate.ProcessedPhosphateSampleDataset
        elif args.analyte == "sulfate":
            estimation_func_class = sulfate.SulfateEstimationFunction
            sample_dataset_class = sulfate.SulfateSampleDataset
            processed_sample_dataset_class = sulfate.ProcessedSulfateSampleDataset
        elif args.analyte == "bisulfite":
            estimation_func_class = bisulfite2d.Bisulfite2DEstimationFunction
            sample_dataset_class = bisulfite2d.Bisulfite2DSampleDataset
            processed_sample_dataset_class = bisulfite2d.ProcessedBisulfite2DSampleDataset
        elif args.analyte == "iron2":
            estimation_func_class = iron2.Iron2EstimationFunction
            sample_dataset_class = iron2.Iron2SampleDataset
            processed_sample_dataset_class = iron2.ProcessedIron2SampleDataset
        # elif args.analyte == "iron32d":
        #     estimation_func_class = iron32d.Iron3EstimationFunction
        #     sample_dataset_class = iron32d.Iron3SampleDataset
        #     processed_sample_dataset_class = iron32d.ProcessedIron3SampleDataset
        elif args.analyte == "iron_oxid":
            estimation_func_class = iron_oxid.IronOxidEstimationFunction
            sample_dataset_class = iron_oxid.IronOxidSampleDataset
            processed_sample_dataset_class = iron_oxid.ProcessedIronOxidSampleDataset
        # elif args.analyte == "iron22d":
        #     estimation_func_class = iron22d.Iron2EstimationFunction
        #     sample_dataset_class = iron22d.Iron2SampleDataset
        #     processed_sample_dataset_class = iron22d.ProcessedIron2SampleDataset
        # elif args.analyte == "iron32d":
        #     estimation_func_class = iron32d.Iron3EstimationFunction
        #     sample_dataset_class = iron32d.Iron3SampleDataset
        #     processed_sample_dataset_class = iron32d.ProcessedIron3SampleDataset
        elif args.analyte == "redox":
            estimation_func_class = redox.RedoxEstimationFunction
            sample_dataset_class = redox.RedoxSampleDataset
            processed_sample_dataset_class = redox.ProcessedRedoxSampleDataset
        elif args.analyte == "ph":
            estimation_func_class = ph.PhEstimationFunction
            sample_dataset_class = ph.PhSampleDataset
            processed_sample_dataset_class = ph.ProcessedPhSampleDataset
        elif args.analyte == "suspended":
            estimation_func_class = suspended.SuspendedEstimationFunction
            sample_dataset_class = suspended.SuspendedSampleDataset
            processed_sample_dataset_class = suspended.ProcessedSuspendedSampleDataset
        else:
            raise NotImplementedError
        # Load PCA statistics.
        pca_stats: Dict[str, np.ndarray] = dict(np.load(args.pca_stats)) if args.pca_stats is not None else dict()
        # Load white balance statistics.
        whitebalance_stats = np.load(args.whitebalance_stats)
        # Make the estimation function object.
        estimation_func = estimation_func_class(checkpoint=args.checkpoint).to(args.device)
        estimation_func.eval()
        # Make the dataset object.
        dataset = sample_dataset_class(base_dirs=args.samples_base_dirs, progress_bar=True, skip_blank_samples=args.skip_blank_samples, skip_incomplete_samples=args.skip_incomplete_samples, skip_inference_sample=args.skip_inference_sample, skip_training_sample=args.skip_training_sample, verbose=True)
        if args.use_expanded_set:
            dataset = ExpandedSampleDataset(dataset, progress_bar=True)
        dataset = processed_sample_dataset_class(dataset, root_dir, num_augmented_samples=0, progress_bar=True, **pca_stats)
        # Create figure to show the sample, masks, and extra data.
        fig, ax = plt.subplots(5, 5, figsize=(13, 20))
        # Compute extra data.
        for processed_sample in tqdm(iter(dataset), desc="Computing extra data to report and writing PDFs" if args.write_pdfs else "Computing extra data to report", total=len(dataset), leave=True):
            sample = processed_sample.sample
            sample_prefix = processed_sample.sample_prefix
            pdf_path = os.path.join(args.results_base_dir, pdf_relative_path(sample))
            # Cleanup.
            for row in range(ax.shape[0]):
                for col in range(ax.shape[1]):
                    ax[row, col].clear()
                    ax[row, col].axis("off")
            # Get the image and some masks of the sample.
            sample_bgr_img = processed_sample.sample_bgr_image
            sample_rgb_img = cv2.cvtColor(sample_bgr_img, cv2.COLOR_BGR2RGB)
            sample_lab_img = processed_sample.sample_lab_image
            sample_analyte_msk = processed_sample.sample_analyte_mask
            sample_grid_msk = processed_sample.sample_grid_mask
            # Compute the cropped version of the image of the sample.
            cropped_sample_bgr_img_path = os.path.join(root_dir, f'{sample_prefix}-cropped.png')
            if os.path.isfile(cropped_sample_bgr_img_path):
                cropped_sample_bgr_img = cv2.imread(cropped_sample_bgr_img_path, cv2.IMREAD_COLOR)
                if cropped_sample_bgr_img is None:
                    raise RuntimeError(f'Can\'t load the file "{cropped_sample_bgr_img_path}"')
            else:
                cropped_sample_bgr_img, = crop_analyte((sample_bgr_img,), sample_analyte_msk)
                if not cv2.imwrite(cropped_sample_bgr_img_path, cropped_sample_bgr_img):
                    raise RuntimeError(f'Can\'t save the file "{cropped_sample_bgr_img_path}"')
            # Compute mean color of the sample.
            sample_mean_lab = np.mean(sample_lab_img[sample_analyte_msk], axis=0)
            sample_mean_rgb = lab_to_rgb(sample_mean_lab)
            # Compute absolute and relative errors of the value estimated by the app.
            true_value = sample["correctedTheoreticalValue"]
            estimated_value = sample["estimatedValue"]
            if estimated_value is not None and true_value is not None:
                estimated_value_absolute_error = abs(estimated_value - true_value)
                if true_value != 0.0:
                    estimated_value_relative_error = estimated_value_absolute_error / true_value
                else:
                    estimated_value_relative_error = None
            else:
                estimated_value_absolute_error = None
                estimated_value_relative_error = None
            # Handle the blank sample assigned to the actual sample.
            if processed_sample.has_valid_blank:
                blank = sample["blank"]
                blank_prefix = processed_sample.blank_prefix
                # Perform offline estimation and compute absolute and relative errors.
                if args.compute_offline_estimation:
                    calibrated_pmf = processed_sample.calibrated_pmf
                    offline_estimated_value = estimation_func(calibrated_pmf=torch.as_tensor(calibrated_pmf, dtype=torch.float32, device=args.device).unsqueeze(0)).item()
                    if true_value is not None:
                        offline_estimated_value_absolute_error = abs(offline_estimated_value - true_value)
                        if true_value != 0.0:
                            offline_estimated_value_relative_error = offline_estimated_value_absolute_error / true_value
                        else:
                            offline_estimated_value_relative_error = None
                    else:
                        offline_estimated_value_absolute_error = None
                        offline_estimated_value_relative_error = None
                else:
                    offline_estimated_value = None
                    offline_estimated_value_absolute_error = None
                    offline_estimated_value_relative_error = None
                # Update sample's summary.
                if args.write_pdfs and not os.path.isfile(pdf_path):
                    # Get the image and some masks of the blank sample.
                    blank_bgr_img = processed_sample.blank_bgr_image
                    blank_rgb_img = cv2.cvtColor(blank_bgr_img, cv2.COLOR_BGR2RGB)
                    blank_lab_img = processed_sample.blank_lab_image
                    blank_analyte_msk = processed_sample.blank_analyte_mask
                    # Compute mean color of the blank sample.
                    blank_mean_lab = np.mean(blank_lab_img[blank_analyte_msk], axis=0)
                    blank_mean_rgb = lab_to_rgb(blank_mean_lab)
                    # Get white color of the blank sample.
                    blank_lab_white = processed_sample.blank_lab_white
                    blank_rgb_white = lab_to_rgb(blank_lab_white)
                    # Draw image and masks of the blank sample.
                    draw_image_and_masks(
                        ax[1, :],
                        title="Amostra de Zeragem",
                        name=blank["name"] if blank is not None else "Sem Nome",
                        solutions=None,
                        source_stock_name=None,
                        standard_volume=blank["standardVolume"] if blank is not None else 0.0,
                        used_volume=blank["usedVolume"] if blank is not None else 0.0,
                        volume_unit=blank["volumeUnit"] if blank is not None else sample["volumeUnit"],
                        true_value=None,
                        estimated_value=None,
                        offline_estimated_value=None,
                        value_unit=blank["valueUnit"] if blank is not None else sample["valueUnit"],
                        notes=blank["notes"] if blank is not None else "JSON não associado. Apenas a imagem está disponível.",
                        analyst_name=blank["analystName"] if blank is not None else sample["analystName"],
                        datetime=blank["datetime"] if blank is not None else None,
                        rgb_img=blank_rgb_img,
                        bright_msk=processed_sample.blank_bright_mask,
                        grid_msk=processed_sample.blank_grid_mask,
                        analyte_msk=blank_analyte_msk,
                        lab_white=blank_lab_white,
                        rgb_white=blank_rgb_white,
                        mean_lab=blank_mean_lab,
                        mean_rgb=blank_mean_rgb,
                    )
                    # Get PMFs.
                    sample_pmf = processed_sample.sample_pmf
                    blank_pmf = processed_sample.blank_pmf
                    calibrated_pmf = processed_sample.calibrated_pmf
                    sample_img_to_pmf = processed_sample.sample_image_to_pmf
                    blank_img_to_pmf = processed_sample.blank_image_to_pmf
                    # Compute activation maps.
                    sample_pmf_activation, blank_pmf_activation, calibrated_pmf_activation = relative_contribution_at_pmfs(estimation_func, sample_pmf=sample_pmf, blank_pmf=blank_pmf, calibrated_pmf=calibrated_pmf, device=args.device)
                    sample_img_activation = relative_contribution_at_img(bgr_img=sample_bgr_img, relative_contribution_pmf=sample_pmf_activation, img_to_pmf=sample_img_to_pmf)
                    blank_img_activation = relative_contribution_at_img(bgr_img=blank_bgr_img, relative_contribution_pmf=blank_pmf_activation, img_to_pmf=blank_img_to_pmf)
                    # Compute the cropped version of the activation image of the sample.
                    cropped_sample_img_activation, cropped_sample_analyte_msk = crop_analyte((sample_img_activation, sample_analyte_msk), sample_analyte_msk)
                    cropped_sample_analyte_alpha = cropped_sample_analyte_msk.astype(np.float32)
                    # Compute the cropped version of the images of the blank sample.
                    cropped_blank_bgr_img_path = os.path.join(root_dir, f'{blank_prefix}-cropped.png')
                    if os.path.isfile(cropped_blank_bgr_img_path):
                        cropped_blank_bgr_img = cv2.imread(cropped_blank_bgr_img_path, cv2.IMREAD_COLOR)
                        if cropped_blank_bgr_img is None:
                            raise RuntimeError(f'Can\'t load the file "{cropped_blank_bgr_img_path}"')
                        cropped_blank_img_activation, cropped_blank_analyte_msk = crop_analyte((blank_img_activation, blank_analyte_msk), blank_analyte_msk)
                    else:
                        cropped_blank_bgr_img, cropped_blank_img_activation, cropped_blank_analyte_msk = crop_analyte((blank_bgr_img, blank_img_activation, blank_analyte_msk), blank_analyte_msk)
                        if not cv2.imwrite(cropped_blank_bgr_img_path, cropped_blank_bgr_img):
                            raise RuntimeError(f'Can\'t save the file "{cropped_blank_bgr_img_path}"')
                    cropped_blank_analyte_alpha = cropped_blank_analyte_msk.astype(np.float32)
                    # Draw distributions.
                    vmax = max(sample_pmf.max().item(), blank_pmf.max().item(), calibrated_pmf.max().item())
                    draw_distribution(ax[2, 0], "Distribuição da\nAmostra", sample_pmf, cmap="cividis", vmin=0.0, vmax=vmax)
                    draw_distribution(ax[2, 1], "Distribuição da\nAmostra de Zeragem", blank_pmf, cmap="cividis", vmin=0.0, vmax=vmax)
                    draw_distribution(ax[2, 2], "Distribuição\nPadronizada", calibrated_pmf, cmap="cividis", vmin=0.0, vmax=vmax)
                    # Draw activation maps.
                    vmax = max(sample_pmf_activation.max().item(), blank_pmf_activation.max().item())
                    draw_distribution(ax[3, 0], "Ativação na\nDistribuição da Amostra", sample_pmf_activation, cmap="cividis", vmin=0.0, vmax=vmax)
                    draw_distribution(ax[3, 1], "Ativação na\nDistribuição da\nAmostra de Zeragem", blank_pmf_activation, cmap="cividis", vmin=0.0, vmax=vmax)
                    vmax = calibrated_pmf_activation.max().item()
                    draw_distribution(ax[3, 2], "Ativação na\nDistribuição\nPadronizada", calibrated_pmf_activation, cmap="cividis", vmin=0.0, vmax=vmax)
                    vmax = max(cropped_sample_img_activation.max().item(), cropped_blank_img_activation.max().item())
                    ax[2, 3].set_title("ROI da\nAmostra")
                    ax[2, 3].imshow(np.concatenate((cv2.cvtColor(cropped_sample_bgr_img, cv2.COLOR_BGR2RGB), (255 * cropped_sample_analyte_alpha[:, :, np.newaxis]).astype(np.uint8)), axis=2))
                    ax[3, 3].set_title("Ativação na\nROI da Amostra")
                    ax[3, 3].imshow(cropped_sample_img_activation, alpha=cropped_sample_analyte_alpha, cmap="cividis", norm=colors.LogNorm(vmin=sys.float_info.epsilon, vmax=vmax, clip=True))
                    ax[2, 4].set_title("\nROI da\nAmostra de Zeragem")
                    ax[2, 4].imshow(np.concatenate((cv2.cvtColor(cropped_blank_bgr_img, cv2.COLOR_BGR2RGB), (255 * cropped_blank_analyte_alpha[:, :, np.newaxis]).astype(np.uint8)), axis=2))
                    ax[3, 4].set_title("Ativação na\nROI da\nAmostra de Zeragem")
                    ax[3, 4].imshow(cropped_blank_img_activation, alpha=cropped_blank_analyte_alpha, cmap="cividis", norm=colors.LogNorm(vmin=sys.float_info.epsilon, vmax=vmax, clip=True))
            else:
                # Set offline estimated value to None.
                offline_estimated_value = None
                offline_estimated_value_absolute_error = None
                offline_estimated_value_relative_error = None
                # Update sample's summary.
                if args.write_pdfs and not os.path.isfile(pdf_path):
                    sample_pmf = processed_sample.sample_pmf
                    vmax = sample_pmf.max().item()
                    draw_distribution(ax[2, 0], "Distribuição da\nAmostra", sample_pmf, cmap="cividis", vmin=0.0, vmax=vmax)
            # Update sample's summary.
            if args.write_pdfs and not os.path.isfile(pdf_path):
                # Get white color of the sample.
                sample_lab_white = processed_sample.sample_lab_white
                sample_rgb_white = lab_to_rgb(sample_lab_white)
                # Draw image and masks of the sample.
                draw_image_and_masks(
                    ax[0, :],
                    "Amostra",
                    name=sample["name"],
                    solutions=sample["auxiliarySolutions"],
                    source_stock_name=sample["sourceStock"]["name"],
                    standard_volume=sample["standardVolume"],
                    used_volume=sample["usedVolume"],
                    volume_unit=sample["volumeUnit"],
                    true_value=true_value,
                    estimated_value=estimated_value,
                    offline_estimated_value=offline_estimated_value,
                    value_unit=sample["valueUnit"],
                    notes=sample["notes"],
                    analyst_name=sample["analystName"],
                    datetime=sample["datetime"],
                    rgb_img=sample_rgb_img,
                    bright_msk=processed_sample.sample_bright_mask,
                    grid_msk=sample_grid_msk,
                    analyte_msk=sample_analyte_msk,
                    lab_white=sample_lab_white,
                    rgb_white=sample_rgb_white,
                    mean_lab=sample_mean_lab,
                    mean_rgb=sample_mean_rgb,
                )
                # Save to PDF.
                os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
                plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
            # Assign extra data to the sample.
            sample["extra"] = {
                "croppedSampleFileName": cropped_sample_bgr_img_path,
                "meanLab": sample_mean_lab,
                "meanRGB": sample_mean_rgb,
                "confidenceInWhiteBalance": estimate_confidence_in_whitebalance(lab_img=sample_lab_img, grid_msk=sample_grid_msk, whitebalance_stats=whitebalance_stats),
                "estimatedValueAbsoluteError": estimated_value_absolute_error,
                "estimatedValueRelativeError": estimated_value_relative_error,
                "offlineEstimatedValue": offline_estimated_value,
                "offlineEstimatedValueAbsoluteError": offline_estimated_value_absolute_error,
                "offlineEstimatedValueRelativeError": offline_estimated_value_relative_error,
            }
        plt.close(fig)
        # Write workbook.
        write_workbook(args.results_base_dir, dataset, estimation_func.net, args.write_pdfs)
    finally:
        if args.clear_temp_files:
            shutil.rmtree(root_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    # Set network model arguments.
    group = parser.add_argument_group("model arguments")
    group.add_argument("--analyte", metavar="TYPE", choices=ANALYTE_CHOICES, type=str, default=DEFAULT_ANALYTE)
    group.add_argument("--checkpoint", metavar="FILEPATH", type=str)
    group.add_argument("--pca_stats", metavar="FILEPATH", type=str)
    group.add_argument("--whitebalance_stats", metavar="FILEPATH", type=str, default=WHITEBALANCE_STATS)
    group.add_argument("--device", metavar="NAME", type=str, default=DEFAULT_DEVICE_NAME)
    # Set sample arguments.
    group = parser.add_argument_group("sample arguments")
    group.add_argument("--samples_base_dirs", nargs="+", default=[])
    switch = group.add_mutually_exclusive_group()
    switch.add_argument("--use_expanded_set", dest="use_expanded_set", action="store_true")
    switch.add_argument("--dont_use_expanded_set", dest="use_expanded_set", action="store_false")
    switch.set_defaults(use_expanded_set=DEFAULT_USE_EXPANDED_SET)
    switch = group.add_mutually_exclusive_group()
    switch.add_argument("--skip_blank_samples", dest="skip_blank_samples", action="store_true")
    switch.add_argument("--dont_skip_blank_samples", dest="skip_blank_samples", action="store_false")
    switch.set_defaults(skip_blank_samples=DEFAULT_SKIP_BLANK_SAMPLES)
    switch = group.add_mutually_exclusive_group()
    switch.add_argument("--skip_incomplete_samples", dest="skip_incomplete_samples", action="store_true")
    switch.add_argument("--dont_skip_incomplete_samples", dest="skip_incomplete_samples", action="store_false")
    switch.set_defaults(skip_incomplete_samples=DEFAULT_SKIP_INCOMPLETE_SAMPLES)
    switch = group.add_mutually_exclusive_group()
    switch.add_argument("--skip_inference_sample", dest="skip_inference_sample", action="store_true")
    switch.add_argument("--dont_skip_inference_sample", dest="skip_inference_sample", action="store_false")
    switch.set_defaults(skip_inference_sample=DEFAULT_SKIP_INFERENCE_SAMPLES)
    switch = group.add_mutually_exclusive_group()
    switch.add_argument("--skip_training_sample", dest="skip_training_sample", action="store_true")
    switch.add_argument("--dont_skip_training_sample", dest="skip_training_sample", action="store_false")
    switch.set_defaults(skip_training_sample=DEFAULT_SKIP_TRAINING_SAMPLES)
    # Set output arguments.
    group = parser.add_argument_group("output arguments")
    group.add_argument("--results_base_dir", metavar="PATH", type=str)
    switch = group.add_mutually_exclusive_group()
    switch.add_argument("--clear_temp_files", dest="clear_temp_files", action="store_true")
    switch.add_argument("--dont_clear_temp_files", dest="clear_temp_files", action="store_false")
    switch.set_defaults(clear_temp_files=DEFAULT_CLEAR_TEMP_FILES)
    switch = group.add_mutually_exclusive_group()
    switch.add_argument("--compute_offline_estimation", dest="compute_offline_estimation", action="store_true")
    switch.add_argument("--dont_compute_offline_estimation", dest="compute_offline_estimation", action="store_false")
    switch.set_defaults(compute_offline_estimation=DEFAULT_COMPUTE_OFFLINE_ESTIMATION)
    switch = group.add_mutually_exclusive_group()
    switch.add_argument("--write_pdfs", dest="write_pdfs", action="store_true")
    switch.add_argument("--dont_write_pdfs", dest="write_pdfs", action="store_false")
    switch.set_defaults(write_pdfs=DEFAULT_WRITE_PDFS)
    # Parse arguments.
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = DEFAULT_CHECKPOINT.get(args.analyte, None)
    if args.pca_stats is None:
        args.pca_stats = DEFAULT_PCA_STATS.get(args.analyte, None)
    if len(args.samples_base_dirs) == 0:
        args.samples_base_dirs = DEFAULT_SAMPLES_BASE_DIRS[args.analyte]
    # Produce one report for each base dir of samples.
    results_base_dir = args.results_base_dir if args.results_base_dir is not None else os.path.join(os.path.dirname(__file__), "results", args.analyte)
    for samples_base_dir in args.samples_base_dirs:
        args.samples_base_dirs = [samples_base_dir]
        args.results_base_dir = os.path.join(results_base_dir, os.path.basename(samples_base_dir))
        main(args)
