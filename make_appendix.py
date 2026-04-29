#!/usr/bin/env python3
"""Build trimmed Appendix — Code.docx from curated code excerpts."""

from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT = Path.home() / "Downloads" / "Appendix — Code.docx"
REPO = "https://github.com/Nonsense2018/PredictionAP"

# ── Trimmed excerpts ──────────────────────────────────────────────────────────

EXCERPT_A = f"""\
# Excerpt — full code available at {REPO}

AIRNOW_ENDPOINT = "https://www.airnowapi.org/aq/data/"

def build_params(api_key, latitude, longitude, day, bbox_half_size=0.25):
    min_lon = longitude - bbox_half_size
    min_lat = latitude - bbox_half_size
    max_lon = longitude + bbox_half_size
    max_lat = latitude + bbox_half_size
    return {{
        "startDate": f"{{day.isoformat()}}T00",
        "endDate":   f"{{day.isoformat()}}T23",
        "parameters": "PM25",
        "BBOX": f"{{min_lon}},{{min_lat}},{{max_lon}},{{max_lat}}",
        "dataType": "B",
        "format": "application/json",
        "verbose": "1",
        "monitorType": "0",
        "includerawconcentrations": "1",
        "API_KEY": api_key,
    }}

def fetch_county_records_for_day(api_key, county, latitude, longitude, day,
                                  bbox_half_size=0.25):
    params = build_params(api_key, latitude, longitude, day, bbox_half_size)
    response = requests.get(AIRNOW_ENDPOINT, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        return pd.DataFrame()
    frame = pd.DataFrame(payload)
    frame["county"] = county
    frame["request_date"] = day.isoformat()
    return frame

for row in centroids.itertuples(index=False):
    county = str(row.county)
    lat    = float(row.latitude)
    lon    = float(row.longitude)
    for day in iter_dates(start_date, end_date):
        county_frame = fetch_county_records_for_day(api_key, county, lat, lon, day)
        if not county_frame.empty:
            county_frames.append(county_frame)
"""

EXCERPT_B = f"""\
# Excerpt — full code available at {REPO}

OPEN_METEO_ARCHIVE_ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"

def fetch_county_weather_range(county, latitude, longitude, start_date, end_date):
    params = {{
        "latitude":   latitude,
        "longitude":  longitude,
        "start_date": start_date.isoformat(),
        "end_date":   end_date.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
        "timezone": "UTC",
    }}
    response = requests.get(OPEN_METEO_ARCHIVE_ENDPOINT, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    frame = pd.DataFrame(payload.get("daily", {{}}))
    frame["county"] = county
    frame["temperature_2m_mean"] = (
        frame["temperature_2m_max"] + frame["temperature_2m_min"]
    ) / 2.0
    frame = frame.rename(columns={{"time": "date"}})
    return frame

columns = [
    "county", "date",
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "wind_speed_10m_max",
]
weather = weather[columns].drop_duplicates(subset=["county", "date"])
"""

EXCERPT_C = f"""\
# Excerpt — full code available at {REPO}

EONET_EVENTS_ENDPOINT = "https://eonet.gsfc.nasa.gov/api/v3/events"
EARTH_RADIUS_KM  = 6371.0
DEFAULT_RADIUS_KM = 150.0

def haversine_km(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def fetch_wildfire_events(start_date, end_date):
    params = {{
        "status":   "all",
        "category": "wildfires",
        "start":    start_date.isoformat(),
        "end":      end_date.isoformat(),
        "limit":    2000,
    }}
    response = requests.get(EONET_EVENTS_ENDPOINT, params=params, timeout=60)
    response.raise_for_status()
    return response.json().get("events", [])

for centroid in centroids.itertuples(index=False):
    county = str(centroid.county)
    c_lat  = float(centroid.latitude)
    c_lon  = float(centroid.longitude)
    for day in all_dates:
        day_events = events[events["date"] == day]
        count     = 0
        distances = []
        for event in day_events.itertuples(index=False):
            dist = haversine_km(c_lat, c_lon,
                                float(event.latitude), float(event.longitude))
            distances.append(dist)
            if dist <= DEFAULT_RADIUS_KM:
                count += 1
        rows.append({{
            "county":                county,
            "date":                  day,
            "fire_event_count_radius": count,
            "min_fire_distance_km":  min(distances) if distances else pd.NA,
            "fire_radius_km":        DEFAULT_RADIUS_KM,
        }})
"""

EXCERPT_D = f"""\
# Excerpt — full code available at {REPO}

EXCEEDANCE_THRESHOLD = 100

grouped = df.groupby("county", group_keys=False)

df["aqi_lag_1"] = grouped["aqi_mean"].shift(1)
df["aqi_lag_2"] = grouped["aqi_mean"].shift(2)
df["aqi_lag_3"] = grouped["aqi_mean"].shift(3)

df["aqi_roll3_mean"] = (
    grouped["aqi_mean"].shift(1).rolling(window=3).mean()
    .reset_index(level=0, drop=True)
)
df["aqi_roll7_mean"] = (
    grouped["aqi_mean"].shift(1).rolling(window=7).mean()
    .reset_index(level=0, drop=True)
)

df["target_next_day_aqi"]        = grouped["aqi_mean"].shift(-1)
df["target_next_day_exceedance"] = (
    df["target_next_day_aqi"] >= EXCEEDANCE_THRESHOLD
).astype("float")
"""

EXCERPT_E = f"""\
# Excerpt — full code available at {REPO}

EXCEEDANCE_THRESHOLD = 100

train_df = model_frame[
    (model_frame["date"] >= train_start) & (model_frame["date"] <= train_end)
]
val_df = model_frame[
    (model_frame["date"] >= val_start) & (model_frame["date"] <= val_end)
]
test_df = model_frame[
    (model_frame["date"] >= test_start) & (model_frame["date"] <= test_end)
]

cls_models = [
    ("logistic", Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        )),
    ])),
    ("random_forest", RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    )),
]

persistence_cls = (test_df["aqi_lag_1"].values >= EXCEEDANCE_THRESHOLD).astype(int)

for name, model in cls_models:
    model.fit(x_train, y_train_cls)
    y_pred  = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    metrics = {{
        "recall":  recall_score(y_test_cls, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test_cls, y_proba),
    }}

persistence_metrics = {{
    "recall":  recall_score(y_test_cls, persistence_cls, zero_division=0),
}}
"""

# ── Section definitions ───────────────────────────────────────────────────────

SECTIONS = [
    (
        "Appendix A — Air Quality Data Collection",
        "This script queries the EPA AirNow API to retrieve daily AQI and PM2.5 readings for each of the eight SJV counties using a \u00b10.25\u00b0 bounding box centered on each county centroid.",
        EXCERPT_A,
    ),
    (
        "Appendix B — Meteorological Data Collection",
        "This script queries the Open-Meteo Historical Weather API to retrieve daily maximum temperature, minimum temperature, total precipitation, and maximum wind speed for each county.",
        EXCERPT_B,
    ),
    (
        "Appendix C — Wildfire Proximity Data Collection",
        "This script queries the NASA EONET v3 API to retrieve wildfire event locations and computes the active fire count and minimum distance to fire within a 150-kilometer radius for each county-day observation.",
        EXCERPT_C,
    ),
    (
        "Appendix D — Feature Engineering",
        "This script merges the three data sources by county and date and constructs all predictive features including AQI lag features, three-day and seven-day rolling means, and the hazardous exceedance target.",
        EXCERPT_D,
    ),
    (
        "Appendix E — Model Training and Evaluation",
        "This script trains and evaluates Logistic Regression, Random Forest, and a Persistence Baseline using a chronological train/validation/test split, with class-weight balancing and recall and ROC-AUC as evaluation metrics.",
        EXCERPT_E,
    ),
]


# ── Build document ────────────────────────────────────────────────────────────

def add_page_number(doc):
    section = doc.sections[0]
    footer  = section.footer
    para    = footer.paragraphs[0]
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = para.add_run()
    for tag, text in [("w:fldChar", None), ("w:instrText", "PAGE"), ("w:fldChar", None)]:
        el = OxmlElement(tag)
        if text:
            el.text = text
        else:
            el.set(qn("w:fldCharType"), "begin" if not run._r.findall(f"{{{qn('w:fldChar').split('}')[0][1:]}}}fldChar") else "end")
        run._r.append(el)


def add_page_number_v2(doc):
    section = doc.sections[0]
    footer  = section.footer
    para    = footer.paragraphs[0]
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = para.add_run()

    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    run._r.append(begin)

    instr = OxmlElement("w:instrText")
    instr.text = "PAGE"
    run._r.append(instr)

    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(end)


def main():
    doc = Document()

    for sec in doc.sections:
        sec.top_margin    = Inches(1)
        sec.bottom_margin = Inches(1)
        sec.left_margin   = Inches(1)
        sec.right_margin  = Inches(1)

    add_page_number_v2(doc)

    for idx, (title, desc, code) in enumerate(SECTIONS):
        if idx > 0:
            doc.add_page_break()

        h = doc.add_paragraph()
        r = h.add_run(title)
        r.bold = True
        r.font.name = "Times New Roman"
        r.font.size = Pt(12)

        d = doc.add_paragraph()
        r = d.add_run(desc)
        r.font.name = "Times New Roman"
        r.font.size = Pt(12)

        doc.add_paragraph()

        for line in code.splitlines():
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after  = Pt(0)
            r = p.add_run(line if line.strip() else " ")
            r.font.name = "Courier New"
            r.font.size = Pt(10)

    doc.save(OUT)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
