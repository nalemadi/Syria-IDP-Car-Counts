"""
Partial coverage correction for satellite imagery car counts.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BETA = 0.0                                   # selected on the validation experiment
BASELINE_END = datetime(yyyy, mm, dd)         # images after this date are corrected

SEAMLINE_PATH = "sat_imgs/{}/{}/SEAMLINES_shape.geojson"
CAR_DETECTIONS_PATH = "car_coords/car_coords_{}_0.55.csv"
BASELINE_GRID_DIR = Path("prewar_baseline_cars_count_grids")
AREA_COVERAGE_FILE = "area_coverage.txt"
TIMESTAMP_FILE = "timestamp_dict.json"
OUTPUT_FILE = "output/partial_coverage_corrected_counts.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_json_data(filename):
    with open(filename) as f:
        return json.load(f)


def read_area_coverage(fname):
    """
    Read '<city>/<date>,<coverage>' lines into {'<city>_<date>': coverage}.
    """
    coverage = {}
    with open(fname) as f:
        for line in f:
            key, value = line.strip().split(",")
            city, date = key.split("/")
            coverage[f"{city}_{date}"] = float(value)
    return coverage


def split_baseline_cars(baseline_gdf, seamline_file):
    """
    Split baseline cars into those inside (visible) and outside (invisible) the covered area defined by the seamline polygons.
    """
    with open(seamline_file) as f:
        seamlines = json.load(f)

    visible, invisible = 0, 0
    for _, cell in baseline_gdf.iterrows():
        for feature in seamlines["features"]:
            if shape(feature["geometry"]).contains(cell.geometry):
                visible += cell.mean_Ncars
            else:
                invisible += cell.mean_Ncars
            break  # only the first seamline polygon is evaluated
    return int(round(visible)), int(round(invisible))


def shrinkage_scaling(c, r, beta=BETA):
    """
    theta = 1 + c^beta * (r - 1)
    """
    return 1 + (c ** beta) * (r - 1)


def corrected_car_count(city, date, detected_cars, baseline_path):
    """
    Corrected total car count for one partially covered image.
    """
    seamline_file = SEAMLINE_PATH.format(city, date)

    baseline_gdf = gpd.read_file(baseline_path)
    baseline_total = int(round(baseline_gdf.mean_Ncars.sum()))
    baseline_visible, baseline_invisible = split_baseline_cars(baseline_gdf, seamline_file)

    c = baseline_visible / baseline_total
    r = len(detected_cars) / baseline_visible if baseline_visible != 0 else 0
    theta = shrinkage_scaling(c, r)

    estimated_invisible = theta * baseline_invisible
    return int(round(estimated_invisible + len(detected_cars)))


def main():
    baseline_paths = {
        p.name.split("_baseline.gpkg")[0]: str(p)
        for p in BASELINE_GRID_DIR.iterdir() if p.is_file()
    }
    acq_dates = read_json_data(TIMESTAMP_FILE)
    area_coverage = read_area_coverage(AREA_COVERAGE_FILE)

    header = ["city", "date", "areaCov", "acq_dates_lst", "car_count", "est_car_count"]
    rows = []
    for city_date, area_cov in area_coverage.items():
        city = "_".join(city_date.split("_")[:-1])
        date = city_date.split("_")[-1]

        if datetime.strptime(date, "%b%Y") <= BASELINE_END:
            continue

        detections_file = CAR_DETECTIONS_PATH.format(city_date)
        if os.path.exists(detections_file):
            detected_cars = pd.read_csv(detections_file)
        else:
            detected_cars = []
        car_count = len(detected_cars)

        if area_cov == 100:
            estimate = car_count  # full coverage: no correction needed
        else:
            estimate = corrected_car_count(city, date, detected_cars, baseline_paths[city])
        rows.append([city, date, area_cov, acq_dates[city_date], car_count, estimate])
        print(city_date, area_cov, car_count, "->", estimate)

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=header).to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()
