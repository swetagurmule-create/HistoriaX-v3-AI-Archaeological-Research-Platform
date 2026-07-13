# HistoriaX-v3-AI-Archaeological-Research-Platform
# HistoriaX v3 — AI Archaeological Research Platform

## What was Improved

### Model 1 — GIS / Archaeological Prediction (historical_ai_system.zip)

**5 Critical Bugs Fixed:**

| # | Bug | Fix |
|---|-----|-----|
| BUG-1 | Single `LabelEncoder` reused across all 3 columns (Soil, Vegetation, Target) → labels corrupt | Separate encoder per column: `soil_enc`, `veg_enc`, `target_enc` |
| BUG-2 | Encoders never saved to disk → reload fails / wrong label mapping | All 3 encoders now saved as `.pkl` files |
| BUG-3 | No `class_weight` → High class (1.9% of data) almost never predicted | `class_weight="balanced"` + SMOTE oversampling |
| BUG-4 | `test_prediction.py` sends `{latitude, longitude, elevation}` but model was trained on 7 different features | Test file fixed to use correct feature columns |
| BUG-5 | `train_test_split` without `stratify=y` → evaluation misleadingly high | `stratify=y` added |

**New Improvements:**
- **6 Engineered Features:** `River_Prox_Score`, `Coast_Prox_Score`, `Water_Access`, `Elev_Suit`, `Climate_Score`, `Log_River/Coast/Rain`
- **Ensemble Model:** `RF + GradientBoosting + XGBoost` VotingClassifier with soft probabilities
- **StandardScaler** persisted for consistent inference
- **5-fold stratified cross-validation** reported
- **Feature importance** saved to JSON and shown in UI

### Model 2 — NLP / Manuscript Decoder (modelnlp.zip)

**4 Critical Bugs Fixed:**

| # | Bug | Fix |
|---|-----|-----|
| BUG-1 | `to_knowledge_graph()` only looks up `entity_ids` by exact text → custom entities (title-cased from `known_entities`) never matched → KG has nodes but no edges | Dual lookup: `id_by_text` + `id_by_lower` |
| BUG-2 | `detect_events()` appends global entity list to every event regardless of sentence boundary → wrong participants/locations | Scoped: only entities whose text appears in the same sentence are linked |
| BUG-3 | Confidence always hard-coded 0.85/0.8/0.9 regardless of evidence strength | Computed from pattern strength and entity count |
| BUG-4 | `extract_relations()` only checks direct dependency children → misses "X ruled **over** Y" (prep chain) | Walk `prep` → `pobj` one level deep |

**New Improvements:**
- **300+ Historical Entities** across Indian, Greek, Roman, Chinese, Middle Eastern, Mesoamerican cultures
- **Temporal Reference Extraction:** BC/AD, century, era patterns
- **Geographic Coordinates:** 80 ancient sites with lat/lon for map visualization
- **Co-occurrence Edges:** Entities mentioned together in same sentence get co-mentioned edges
- **Batch Processing:** Thread pool for parallel manuscript processing

### Model 3 — Vision / Artifact Detection (vision_model.zip)

The ArtifactPipeline (CLIP + YOLOv8) architecture is solid. Improvements:
- **Adaptive thresholding CV fallback** when CLIP/YOLO unavailable
- **Confidence calibration** for bounding box display
- **Context-aware CLIP prompts** via civilisation context parameter
- **Richer API response** with embedding dimension, OCR readability, word count

---

## Architecture

```
User → Frontend (Next.js 14)
         ↓
    FastAPI Backend (main.py)
    ┌──────────┬──────────────────┬──────────────────────┐
    │          │                  │                      │
    ▼          ▼                  ▼                      ▼
ArtifactPipeline  ImprovedNLPProcessor  GIS Ensemble    Multi-Modal
(CLIP+YOLO)       (spaCy+300 entities)  (RF+GB+XGB)     Synthesis
    │          │                  │                      │
    ▼          ▼                  ▼                      ▼
/detect_artifact /decode_manuscript /predict_archaeology /multimodal_analysis
                                   /generate_heatmap
```

---

## Setup

### Backend

```bash
cd backend

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm     # base NLP model
python -m spacy download en_core_web_lg     # optional: better NER accuracy

# Train the GIS model (fixes all 5 bugs, saves encoders)
python -m models.train_gis

# Start the API server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev   # → http://localhost:3000
```

---

## API Endpoints

| Method | Endpoint | Model | Description |
|--------|----------|-------|-------------|
| POST | `/detect_artifact` | Vision | Upload image → artifact class + bbox + OCR |
| POST | `/decode_manuscript` | NLP | Text → entities + relations + events + KG |
| POST | `/predict_archaeology` | GIS | 7 GIS features → High/Medium/Low + probabilities |
| GET | `/generate_heatmap` | GIS | region + limit → [{lat, lon, weight, label}] |
| POST | `/multimodal_analysis` | All 3 | Text + GIS → combined hypothesis |
| GET | `/known_sites` | — | UNESCO archaeological sites with coords |
| GET | `/status` | — | Model health + accuracy |
| GET | `/schema/archaeology` | — | Valid Soil_Type + Vegetation_Type values |

---

## GIS Feature Engineering

The 6 new derived features that improve prediction accuracy:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `River_Prox_Score` | `exp(-dist_river/10)` | Sites exponentially more likely near rivers |
| `Coast_Prox_Score` | `exp(-dist_coast/200)` | Coastal proximity matters less but still positive |
| `Water_Access` | `3×river + coast` | Combined water signal, river weighted 3× |
| `Elev_Suit` | Piecewise: 0→50m=0.5, 50–500m=1.0, … | Archaeological sites peak at 50–500m elevation |
| `Climate_Score` | `(temp_opt + rain_opt) / 2` | Optimal ~20°C, 300–1500mm rainfall |
| `Log_River/Coast/Rain` | `log1p(x)` | Correct right-skew in distance features |

---

## Knowledge Graph Fix (Visual Before/After)

**Before (Bug):**
```
entity_0: "Ashoka" (custom entity, stored as "Ashoka")
entity_1: "Mauryan Empire" (spaCy entity, stored as "the Mauryan Empire")

Relation: subject="Ashoka", object="the Mauryan Empire"
→ entity_ids.get("Ashoka") = "entity_0"  ✓
→ entity_ids.get("the Mauryan Empire") = None  ✗  ← custom="Mauryan Empire" doesn't match
→ EDGE MISSING
```

**After (Fix):**
```
id_by_text["Ashoka"] = "entity_0"
id_by_lower["ashoka"] = "entity_0"
id_by_text["the Mauryan Empire"] = "entity_1"
id_by_lower["the mauryan empire"] = "entity_1"
id_by_text["Mauryan Empire"] = "entity_2"
id_by_lower["mauryan empire"] = "entity_2"

get_id("the Mauryan Empire") → "entity_1"  ✓
→ EDGE CREATED
```
