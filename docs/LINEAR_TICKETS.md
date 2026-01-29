# Linear Tickets - Analytics Dashboard

## Ticket 1: Analysis Calculation Engine

---

### Title
**Biomechanical Analysis Calculation Engine**

### Description
Implement the core analysis calculation logic that processes raw pose estimation data from MediaPipe and generates meaningful biomechanical insights for pickleball stroke analysis.

### Acceptance Criteria
- [ ] Calculate joint angles from landmark coordinates (hip rotation, shoulder abduction, knee flexion, elbow flexion)
- [ ] Detect injury risks based on angle thresholds:
  - Shoulder overuse: abduction > 140°
  - Poor kinetic chain: hip rotation < 30°
  - Knee stress: flexion < 90° or > 170°
  - Elbow strain: excessive extension during contact
- [ ] Generate risk percentages per session (% of frames with each issue)
- [ ] Calculate overall risk level (low/medium/high) based on aggregated metrics
- [ ] Support stroke-specific thresholds (serve, dink, drive, overhead)
- [ ] Generate frame-by-frame feedback messages
- [ ] Calculate session summary statistics (total frames, FPS, duration)

### Technical Details
**Files involved:**
- `lib/analysis/metrics.ts` - Joint angle calculations
- `lib/analysis/risk.ts` - Risk detection with thresholds
- `lib/analysis/feedback.ts` - Feedback message generation
- `lib/analysis/index.ts` - Main orchestration
- `lib/analysis/types.ts` - TypeScript interfaces

**Key interfaces:**
```typescript
interface FrameMetrics {
  hip_rotation_deg: number;
  right_shoulder_abduction: number;
  right_knee_flexion: number;
  right_elbow_flexion: number;
  wrist_above_waist: boolean;
}

interface InjuryRisk {
  type: string;
  severity: 'low' | 'medium' | 'high';
  angle: number;
  message: string;
  recommendation: string;
}
```

### Labels
`backend`, `analysis`, `core-feature`

### Priority
**High**

### Story Points
8

---
---

## Ticket 2: Analytics Dashboard UI

---

### Title
**Build Analytics Dashboard for User Progress Tracking**

### Description
Create a comprehensive analytics dashboard that displays aggregated user statistics, technique metrics, risk analysis, and personalized drill recommendations based on their stroke analysis history.

### Acceptance Criteria
- [ ] Hero score card showing overall skill score (0-100) with trend indicator
- [ ] Weekly progress bar chart (last 7 days)
- [ ] Technique metrics cards with optimal ranges (Hip Rotation, Shoulder Abduction, Knee Flexion)
- [ ] Performance breakdown by stroke type
- [ ] Risk analysis section with colored progress bars
- [ ] Personalized drill recommendations based on top issues
- [ ] Recent sessions list with quick navigation to player view
- [ ] Responsive design for mobile and desktop
- [ ] Loading and empty states
- [ ] Authentication required (redirect if not signed in)

### Technical Details
**Files to create:**
- `app/strikesense/analytics/page.tsx` - Main dashboard page
- `app/api/analytics/summary/route.ts` - API endpoint for aggregated data

**API Response Structure:**
```typescript
interface AnalyticsSummary {
  totalSessions: number;
  skillScore: number;
  averageRisk: string;
  favoriteStroke: string;
  riskBreakdown: { shoulder_overuse, poor_kinetic_chain, knee_stress };
  trends: Array<{ date, score, sessions }>;
  strokeBreakdown: Array<{ stroke_type, sessions, avg_score, metrics }>;
  metricCards: Array<{ key, label, current, optimal_min, optimal_max, status }>;
  drills: Array<{ name, description, duration, priority }>;
  recentSessions: Array<{ id, stroke_type, created_at, score }>;
}
```

**Data Source:**
- Query `analysis_jobs` table filtered by `user_id` and `status = 'completed'`
- Extract metrics from `result_json` field

### Design Requirements
- Clean, modern UI with neutral color palette
- Black hero card with large skill score
- Color-coded risk indicators (green/yellow/red)
- Status icons for metric cards (✅ good, ➖ warning, ❌ critical)
- Smooth transitions and hover states

### Dependencies
- Ticket 1 (Analysis Calculation Engine) must be complete
- Supabase database with `analysis_jobs` table
- Clerk authentication integration

### Labels
`frontend`, `dashboard`, `feature`

### Priority
**High**

### Story Points
13

---
---

## Ticket 3: Advanced Biomechanics & Velocity Metrics

---

### Title
**Advanced Biomechanics & Velocity Metrics**

### Description
Implement advanced calculation engines to track velocity, acceleration, and posture for deeper injury risk detection and skill analysis. This extends the core analysis engine with more sophisticated kinematic metrics.

### Acceptance Criteria
- [ ] **Velocity Tracking**: Calculate instantaneous wrist and racket hand speed
  - Detect sudden deceleration (Tennis Elbow risk marker)
  - Measure peak velocity at contact point for power estimation
- [ ] **Spine Posture Analysis**: Calculate torso lean angle (hip-midpoint to shoulder-midpoint vector)
  - Flag excessive forward lean (>45°) during dinks (lower back strain risk)
- [ ] **Asymmetry Detection**: Compare left vs. right side metrics
  - Calculate symmetry index for knee flexion and shoulder height
  - Flag imbalances > 20%
- [ ] **Wrist Flexion**: Calculate angle between forearm and hand
  - Detect excessive wrist flexion/extension (tendonitis risk)
- [ ] Update `AnalysisResult` and `FrameMetrics` interfaces to include new fields
- [ ] Add new thresholds to `risk.ts` for these metrics

### Technical Details
**New calculations needed:**
- **Velocity**: `v = (p2 - p1) / time_delta`
- **Acceleration**: `a = (v2 - v1) / time_delta`
- **Torso Angle**: Vector angle relative to vertical Y-axis
- **Symmetry**: `|left - right| / ((left + right) / 2)`

**Files modified:**
- `lib/analysis/metrics.ts`
- `lib/analysis/risk.ts`
- `lib/analysis/types.ts`

### Labels
`backend`, `analysis`, `enhancement`

### Priority
**Medium**

### Story Points
8

---

## Ticket 4: AI Coach Insights Feature

---

### Title
**Implement AI Coach Insights for Personalized Training Feedback**

### Description
Integrate generative AI to provide personalized, qualitative feedback based on quantitative biomechanics data. This feature adds an "AI Coach" element to the dashboard, summarizing performance, highlighting key takeaways, and suggesting focus drills using natural language.

### Acceptance Criteria
- [x] **API Endpoint (`/api/analytics/ai-summary`)**:
  - Accept `AnalyticsSummary` JSON payload
  - Connect to OpenAI API using `gpt-4o-mini`
  - Generate structured JSON response: `{ summary, takeaways[], focus_drill }`
  - Handle errors gracefully (e.g., API timeouts)

- [x] **Dashboard UI Integration**:
  - Add "✨ Get AI Coach Insights" button to the Hero Score Card
  - Ensure button is responsive (centered on mobile, right-aligned on desktop)
  - Display loading state with appropriate animation ("Analyzing your biomechanics...")

- [x] **Insights Modal Component**:
  - Display AI-generated content in a premium, glassmorphic modal
  - Show Summary text with encouragement
  - List 3 Key Takeaways with bullet points
  - Highlight "Next Focus Drill" in a distinct section
  - Responsive design (scrollable content on mobile, fixed header)

### Technical Details
**API Prompt Structure:**
- Role: Professional Pickleball Coach
- Input: Skill Score, Risk Analysis, Trend Data, Stroke Breakdown
- Output Format: Strictly JSON

**New Components:**
- `app/components/dashboard/AICoachModal.tsx`
- `app/api/analytics/ai-summary/route.ts`

### Labels
`frontend`, `backend`, `ai`, `feature`

### Priority
**High**

### Story Points
5
