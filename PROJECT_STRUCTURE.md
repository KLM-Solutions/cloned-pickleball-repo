# 📂 StrikeSense Project Structure

This document provides a detailed overview of the project's file organization and architecture.

## 🏗️ Root Directory

```
pickle-ball-main/
├── app/                    # Next.js App Router directory
├── lib/                    # Shared utilities and helpers
├── models/                 # AI model weights
├── public/                 # Static assets
├── python/                 # Python backend
├── .gitignore             # Git ignore rules
├── CONTRIBUTING.md        # Contribution guidelines
├── README.md              # Project documentation
├── eslint.config.mjs      # ESLint configuration
├── next.config.ts         # Next.js configuration
├── package.json           # Node.js dependencies
├── postcss.config.mjs     # PostCSS configuration
├── tsconfig.json          # TypeScript configuration
├── yolov8n.pt            # YOLOv8 nano model
└── yolov8n-seg.pt        # YOLOv8 segmentation model
```

## 📱 App Directory (`/app`)

### Main Pages

```
app/
├── page.tsx                    # Home page (stroke selection)
├── layout.tsx                  # Root layout
├── globals.css                 # Global styles
└── strikesense/               # Main application flow
    ├── guide/                 # Camera setup guides
    │   └── page.tsx          # Dynamic guide page (6 stroke types)
    ├── upload/               # Video upload
    │   └── page.tsx          # Upload interface
    ├── crop/                 # Player selection
    │   └── page.tsx          # Cropping interface
    ├── processing/           # Analysis processing
    │   └── page.tsx          # Processing status
    └── player/               # Results display
        └── page.tsx          # Analysis results
```

### API Routes

```
app/api/
├── analyze/
│   └── route.ts              # Video analysis endpoint
└── frames_draft/
    └── [filename]/
        └── route.ts          # Frame serving endpoint
```

### Components

```
app/components/
├── dashboard/                # Dashboard components
│   ├── DashboardHeader.tsx
│   ├── StatCard.tsx
│   └── TimelineFeed.tsx
├── setup/                    # Setup wizard
│   └── ProfessionalGuide.tsx
├── LeftSidebar.tsx          # Session info sidebar
├── MainLayout.tsx           # Main app layout
├── RightSidebar.tsx         # Biomechanics sidebar
├── StrokeTypeSelector.tsx   # Stroke type selector
└── VideoPanel.tsx           # Video player panel
```

## 🛠️ Utilities (`/lib`)

```
lib/
├── extractFrame.ts           # Frame extraction from video
└── videoStorage.ts           # IndexedDB video storage
```

## 🤖 AI Models (`/models`)

```
models/
├── osnet_x0_25_ms.pt        # ReID model for tracking
├── yolov8n.pt               # YOLOv8 detection (root)
└── yolov8n-seg.pt           # YOLOv8 segmentation (root)
```

## 🎨 Public Assets (`/public`)

```
public/
├── images/                   # Static images
│   ├── serve-camera-guide.png
│   ├── groundstroke-camera-guide.png
│   ├── dink-camera-guide.png
│   ├── overhead-camera-guide.png
│   ├── footwork-camera-guide.png
│   └── overall-camera-guide.png
├── videos/                   # Sample videos
└── frames/                   # Generated frames (gitignored)
```

## 🐍 Python Backend (`/python`)

```
python/
├── biomechanics/            # Biomechanics analysis
│   ├── __init__.py
│   ├── angles.py           # Joint angle calculations
│   ├── injury_risk.py      # Injury risk assessment
│   └── stroke_classifier.py # Stroke classification
├── track.py                # Main tracking script
├── requirements.txt        # Python dependencies
└── venv/                   # Virtual environment (gitignored)
```

## 🔄 Data Flow

### 1. User Journey

```
Home → Guide → Upload → Crop → Processing → Results
```

### 2. Video Processing Pipeline

```
Upload (IndexedDB) → Frame Extraction → Python Analysis → Results JSON → Display
```

### 3. API Communication

```
Frontend → /api/analyze → Python Script → Results → Frontend
```

## 📦 Key Dependencies

### Frontend
- **Next.js 16**: React framework
- **React 19**: UI library
- **Tailwind CSS 4**: Styling
- **Three.js**: 3D visualizations
- **Lucide React**: Icons

### Backend
- **YOLOv8**: Pose detection
- **DeepOCSORT**: Player tracking
- **OpenCV**: Video processing
- **NumPy/SciPy**: Numerical computations

## 🎯 Page Routes

| Route | Description |
|-------|-------------|
| `/` | Home page with stroke selection |
| `/strikesense/guide?stroke=serve` | Serve camera guide |
| `/strikesense/guide?stroke=groundstroke` | Groundstroke guide |
| `/strikesense/guide?stroke=dink` | Dink guide |
| `/strikesense/guide?stroke=overhead` | Overhead guide |
| `/strikesense/guide?stroke=footwork` | Footwork guide |
| `/strikesense/guide?stroke=overall` | Overall form guide |
| `/strikesense/upload` | Video upload page |
| `/strikesense/crop` | Player selection page |
| `/strikesense/processing` | Analysis processing page |
| `/strikesense/player` | Results display page |

## 🔧 Configuration Files

| File | Purpose |
|------|---------|
| `next.config.ts` | Next.js configuration |
| `tsconfig.json` | TypeScript compiler options |
| `eslint.config.mjs` | Code linting rules |
| `postcss.config.mjs` | PostCSS/Tailwind config |
| `package.json` | Node.js dependencies & scripts |

## 📝 Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview & setup |
| `CONTRIBUTING.md` | Development guidelines |
| `PROJECT_STRUCTURE.md` | This file |

## 🚀 Build Output

```
.next/                        # Next.js build output (gitignored)
node_modules/                 # Dependencies (gitignored)
public/frames/               # Generated frames (gitignored)
python/__pycache__/          # Python cache (gitignored)
```

## 🔒 Security & Privacy

- Video files stored in browser IndexedDB
- Temporary frames deleted after processing
- No data sent to external servers
- All processing done locally

## 📊 State Management

- **Video Storage**: IndexedDB (via `lib/videoStorage.ts`)
- **Component State**: React useState/useRef
- **URL Parameters**: Next.js searchParams
- **Session Data**: Browser sessionStorage

---

Last Updated: 2025-12-15
