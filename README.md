# Experiment 3: Artefact Integration

**Module:** MOD002691 Computing Project
**Task:** Integrate the winning pipeline from Experiment 2 into a working mobile application
**Recognition model:** GPT-5.2 Vision (Pipeline A winner, F1 = 0.90)
**Stack:** FastAPI + PostgreSQL (backend), React Native + Expo SDK 54 (mobile)

## Experiment Thread

| Experiment | Question | Outcome |
|---|---|---|
| [1 - CNN Benchmark](https://github.com/omorros/food-cv-exp1-cnn-comparison) | Which CNN best classifies 14 produce classes? | EfficientNet-B0 (99.75%, 40 MB) |
| [2 - Pipeline Comparison](https://github.com/omorros/food-cv-exp2-pipeline-evaluation) | Which end-to-end pipeline builds an inventory from a photo? | GPT-5.2 Vision (F1 = 0.90, 10x more robust than YOLO) |
| **3 - Artefact Integration** *(this repo)* | Does the winning pipeline work in a real mobile app? | Full-stack system with 23 passing tests |

## System Architecture

```mermaid
graph TD
    subgraph Mobile["Mobile Client (React Native + Expo)"]
        CAM[Camera / Gallery]
        REVIEW[Draft Review Screen]
        INV[Inventory Screen]
    end

    subgraph Backend["FastAPI Backend"]
        AUTH[Auth Router<br/>JWT + bcrypt]
        ING[Ingestion Router<br/>POST /ingest/image]
        DRAFT[Draft Items Router<br/>CRUD + Confirm]
        INVR[Inventory Router<br/>CRUD]
        VIS[GPT-5.2 Vision Client]
        NORM[Category & Unit<br/>Normalisation]
        EXP[Expiry Prediction<br/>Rule-based]
    end

    DB[(PostgreSQL)]
    API[OpenAI API<br/>GPT-5.2]

    CAM -->|"image + JWT"| ING
    ING --> VIS
    VIS -->|"base64 + prompt"| API
    API -->|"JSON items"| VIS
    VIS --> NORM
    NORM --> EXP
    EXP -->|"DraftItems"| DRAFT
    DRAFT -->|"JSON response"| REVIEW
    REVIEW -->|"POST /confirm"| DRAFT
    DRAFT -->|"InventoryItem"| INVR
    INVR -->|"JSON response"| INV
    AUTH --> DB
    DRAFT --> DB
    INVR --> DB
```

## Recognition Pipeline

```mermaid
flowchart LR
    A["Photo"] --> B["GPT-5.2 Vision\nbase64 + structured prompt"]
    B --> C["Raw detection\nname, category,\nquantity, unit"]
    C --> D["Category\nnormalisation"]
    D --> E["Unit\nnormalisation"]
    E --> F["Expiry\nprediction"]
    F --> G["DraftItems\nin DB"]
    G --> H["User review\nconfirm / edit / skip"]
    H --> I["InventoryItems\npersisted"]
```

The GPT-5.2 client (`gpt52_vision.py`) is fully decoupled from the rest of the app. It implements a clean interface that could be swapped for any alternative model without changing routes or mobile code.

**Category normalisation** maps 15 categories (e.g. `bread` to `bakery`, `other` to `None`). **Unit normalisation** enforces 5 valid units (`Pieces`, `Grams`, `Kilograms`, `Milliliters`, `Liters`) and expands abbreviations.

## Draft-to-Inventory Trust Model

All AI outputs land as **DraftItems** (untrusted, nullable fields, confidence scores). The user must confirm before they become **InventoryItems** (trusted, all fields required). No code path bypasses this.

| Property | DraftItem | InventoryItem |
|---|---|---|
| Trust level | Untrusted (AI-generated) | Trusted (user-confirmed) |
| Required fields | Only `name` | All fields |
| Nullable fields | category, quantity, unit, expiry | None |
| Provenance | `source`, `confidence_score`, `notes` | Implicit (user confirmed) |
| Lifecycle | Created by AI or manual entry | Created only via draft confirmation |

## Expiry Prediction

Rule-based lookup mapping `(category, storage_location)` to `(shelf_life_days, confidence)`. Deterministic, transparent, and every prediction includes a reasoning string.

```mermaid
flowchart TD
    INPUT["name, category, storage_location"] --> EXACT{"Exact match?"}
    EXACT -->|"Yes"| R1["Matched rule\ne.g. dairy + fridge = 7d, 0.85"]
    EXACT -->|"No"| STORAGE{"Storage fallback?"}
    STORAGE -->|"Yes"| R2["Storage default\ne.g. fridge = 7d, 0.50"]
    STORAGE -->|"No"| R3["Absolute default\n7d, 0.30"]
    R1 --> OUT["ExpiryPrediction\nexpiry_date + confidence + reasoning"]
    R2 --> OUT
    R3 --> OUT
```

| Category | Fridge | Freezer | Pantry |
|---|---|---|---|
| Dairy | 7d (0.85) | 60d (0.80) | 1d (0.60) |
| Meat | 3d (0.85) | 90d (0.90) | 1d (0.30) |
| Vegetables | 7d (0.75) | 240d (0.80) | 5d (0.70) |
| Fruits | 10d (0.70) | 180d (0.75) | 5d (0.65) |
| Bakery | 7d (0.75) | 90d (0.85) | 5d (0.80) |
| Eggs | 21d (0.90) | 180d (0.70) | 7d (0.60) |

40+ rules in total covering all category-storage combinations.

## Mobile App

```mermaid
flowchart TD
    ROOT["App Launch"] --> CHECK{"Token in\nSecureStore?"}
    CHECK -->|"No"| AUTH["Auth Stack"]
    CHECK -->|"Yes"| TABS["Tab Bar"]
    AUTH --> LOGIN["Login"]
    AUTH --> REG["Register"]
    LOGIN -->|"Success"| TABS
    TABS --> INVENTORY["Inventory Tab"]
    TABS --> SETTINGS["Settings Tab"]
    INVENTORY -->|"FAB"| ADD["Add Item Modal"]
    INVENTORY -->|"Tap/Swipe"| EDIT["Edit Item Sheet"]
    ADD --> SCAN["Scan from Image"]
    ADD --> MANUAL["Add Manually"]
    SCAN --> DETECTED["Detected Items\nconfirm / edit / skip"]
```

| Screen | Features |
|---|---|
| Inventory | Status pills (expired, expiring soon, total), search, category/expiry filters, swipe-to-edit and swipe-to-delete, pull-to-refresh |
| Add Item | Two modes: scan image (camera/gallery, GPT-5.2 detection, per-item confirm/edit/skip, bulk add/discard) or manual entry (name, category dropdown, quantity, calendar for expiry) |
| Edit Item | Bottom sheet with edit fields, consume buttons (25/50/75/100%), delete |
| Settings | Account info, default storage location, logout |

Design system uses warm organic tones: sage `#7C9A82` (primary), terracotta `#D4846B` (accent), cream `#FAF8F5` (background). Four-tier expiry colour coding (red/amber/yellow/green). Haptic feedback on key interactions. Swipe gestures via `react-native-gesture-handler`.

## Repository Structure

```
.
├── app/                                 # FastAPI backend
│   ├── core/
│   │   ├── config.py                    # Environment variable loading
│   │   ├── database.py                  # PostgreSQL + SQLAlchemy setup
│   │   └── security.py                  # JWT (HS256) + bcrypt
│   ├── models/
│   │   ├── user.py                      # User model
│   │   ├── draft_item.py               # Untrusted AI-generated item
│   │   └── inventory_item.py           # Trusted user-confirmed item
│   ├── schemas/
│   │   ├── auth.py                      # Auth request/response schemas
│   │   ├── draft_item.py               # Draft CRUD schemas
│   │   └── inventory_item.py           # Inventory CRUD schemas
│   ├── routers/
│   │   ├── auth.py                      # /auth/register, /login, /me
│   │   ├── ingestion.py                # POST /ingest/image
│   │   ├── draft_items.py             # Draft CRUD + POST /confirm
│   │   └── inventory_items.py         # Inventory CRUD
│   └── services/
│       ├── ingestion/
│       │   ├── gpt52_vision.py         # GPT-5.2 Vision API client
│       │   └── image_ingestion.py      # Orchestrator: detect, normalise, predict
│       └── expiry_prediction/
│           ├── service.py              # Multi-strategy orchestrator
│           └── strategies/
│               ├── base.py             # Abstract strategy + dataclass
│               └── rule_based.py       # Lookup-table strategy (40+ rules)
│
├── mobile/                              # React Native + Expo (TypeScript)
│   ├── app/
│   │   ├── _layout.tsx                 # Root layout with AuthProvider
│   │   ├── (auth)/                     # Login, register screens
│   │   ├── (tabs)/                     # Inventory, settings tabs
│   │   ├── add-item.tsx               # Scan image or manual entry
│   │   └── edit-item.tsx              # Edit / consume / delete
│   ├── components/
│   │   ├── ui/                         # Button, Card, Input, Badge, etc.
│   │   ├── add-item/                   # DetectedList, ManualForm, EditItemModal
│   │   └── inventory/                  # InventoryItemCard, Header, Filters
│   ├── services/
│   │   ├── api.ts                      # REST client with JWT auth
│   │   └── auth.tsx                    # AuthContext + SecureStore
│   ├── theme/index.ts                  # Colours, typography, spacing
│   ├── types/index.ts                  # TypeScript interfaces
│   └── utils/                          # Unit conversion, inventory merging
│
├── tests/                               # pytest
│   ├── conftest.py                     # SQLite test DB, fixtures
│   ├── test_api.py                     # Auth, draft-to-inventory, ingestion
│   ├── test_image_ingestion.py        # GPT-5.2 client, normalisation
│   └── test_expiry_prediction.py      # Rule-based strategy, determinism
│
├── requirements.txt
└── .env                                 # Not committed
```

## Testing

```bash
pytest tests/ -v
```

| Test File | Tests | Layer | Covers |
|---|---|---|---|
| `test_image_ingestion.py` | 9 | Service | Image type detection, GPT-5.2 mocking, category normalisation (15 mappings), unit normalisation, full pipeline orchestration, error handling |
| `test_expiry_prediction.py` | 6 | Service | Rule-based predictions, fallback behaviour, determinism validation, custom purchase dates, case-insensitive matching |
| `test_api.py` | 8 | Integration | Auth flow, JWT rejection, draft-to-inventory promotion with cleanup, inventory deletion, image ingestion endpoint, file type validation, health check |

**23 tests, all passing.** Tests use a file-backed SQLite database (`test.db`) and mock all GPT-5.2 calls. No API key or PostgreSQL needed to run them.

## API Reference

All endpoints except `/auth/register`, `/auth/login`, and `/health` require JWT in `Authorization: Bearer <token>`.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/auth/register` | Create account, returns JWT |
| `POST` | `/auth/login` | Authenticate, returns JWT |
| `GET` | `/auth/me` | Current user profile |
| `POST` | `/api/ingest/image` | Upload photo, GPT-5.2 detects items, creates DraftItems |
| `GET` | `/api/draft-items` | List drafts |
| `POST` | `/api/draft-items` | Create draft manually |
| `PATCH` | `/api/draft-items/{id}` | Update draft |
| `DELETE` | `/api/draft-items/{id}` | Discard draft |
| `POST` | `/api/draft-items/{id}/confirm` | Promote draft to inventory item |
| `GET` | `/api/inventory` | List inventory (sorted by expiry) |
| `PUT` | `/api/inventory/{id}` | Update item |
| `PATCH` | `/api/inventory/{id}/quantity` | Update quantity |
| `DELETE` | `/api/inventory/{id}` | Delete item |
| `GET` | `/health` | Health check |

## Setup

### Prerequisites

- Python 3.10+
- PostgreSQL
- Node.js 18+ and npm
- OpenAI API key with GPT-5.2 access

### Backend

```bash
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS/Linux

pip install -r requirements.txt

# Create .env at project root:
#   DATABASE_URL=postgresql://postgres:PASSWORD@localhost:5432/snapshelf
#   OPENAI_API_KEY=sk-...
#   JWT_SECRET_KEY=your-random-secret
#   JWT_ALGORITHM=HS256
#   ACCESS_TOKEN_EXPIRE_MINUTES=10080

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Tables auto-create on first startup. API docs at `http://localhost:8000/docs`.

### Mobile

```bash
cd mobile
npm install

# Update API base URL in mobile/services/api.ts to your local IP

npx expo start
```

Scan the QR code with Expo Go. Both devices must be on the same network.

### Tests

```bash
pytest tests/ -v
```

No PostgreSQL or API key required.

## Requirements

```
fastapi
uvicorn
sqlalchemy
psycopg2-binary
python-dotenv
Pillow
requests
python-multipart
openai>=1.0.0
python-jose[cryptography]
passlib[bcrypt]
bcrypt==4.0.1
email-validator
```

Mobile dependencies managed via `mobile/package.json` (React Native 0.81.5, Expo SDK 54).

---

*MOD002691 Computing Project.*
