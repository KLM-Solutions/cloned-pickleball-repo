# Analytics Dashboard - Value Explanations

This document explains what each metric and value in the StrikeSense Analytics Dashboard represents.

---

## Hero Score Card

| Value | Description |
|-------|-------------|
| **Skill Score (0-100)** | Overall technique quality calculated as `100 - (average risk percentage × 2)`. Higher is better. A score of 80+ indicates good form. |
| **Sessions** | Total number of completed stroke analyses for this user. |
| **Top Stroke** | The stroke type you've analyzed most often (Serve, Dink, Drive, or Overhead). |
| **Risk Level** | Overall injury risk level (Low/Medium/High) based on how often risky form appears across all sessions. |
| **Trend Arrow (↑/↓)** | Shows if your skill score improved or declined comparing oldest to newest sessions this week. |

---

## Weekly Progress Chart

| Value | Description |
|-------|-------------|
| **Bar Height** | Daily average skill score for that day. Taller bars = better form that day. |
| **Score Number** | The actual skill score percentage for sessions completed that day. |
| **Empty Bar** | No sessions were recorded on that day. |

---

## Technique Metrics Cards

| Metric | What It Measures | Optimal Range | Why It Matters |
|--------|------------------|---------------|----------------|
| **Hip Rotation** | How much your hips rotate during the stroke (in degrees). | 30° - 90° | Proper hip rotation generates power and reduces arm strain. Too little = relying on arm only. |
| **Shoulder Abduction** | How high your arm raises away from your body (in degrees). | 0° - 140° | Raising above 140° repeatedly risks rotator cuff injury ("shoulder impingement"). |
| **Knee Flexion** | Your knee bend angle during the stroke (in degrees). | 120° - 160° | Athletic stance (~130-150°) provides stability. Too bent (<120°) stresses knees. Too straight (>160°) reduces power. |

**Status Icons:**
- ✅ **Good** - Your average is within the optimal range
- ➖ **Warning** - Slightly outside optimal range (10-20° off)
- ❌ **Critical** - Significantly outside optimal range (>20° off)

---

## Performance by Stroke

| Value | Description |
|-------|-------------|
| **Stroke Icon** | The type of stroke (🎾 Serve, 🤏 Dink, 💪 Drive, ⚡ Overhead). |
| **Sessions** | How many times you've analyzed this specific stroke type. |
| **Score %** | Average skill score for this stroke type specifically. |
| **Risk Badge** | Risk level (Low/Medium/High) for this stroke based on detected issues. |

---

## Risk Analysis

| Risk Category | What It Measures | Threshold |
|---------------|------------------|-----------|
| **Shoulder Overuse** | Percentage of frames where shoulder abduction exceeds 140°. | < 10% is good, > 10% is concerning. |
| **Poor Kinetic Chain** | Percentage of frames with insufficient hip rotation (< 30°) during power strokes. | < 15% is acceptable, > 15% needs work. |
| **Knee Stress** | Percentage of frames where knee bend is outside the safe range (too deep < 90° or too straight). | < 10% is good, > 10% is concerning. |

**Bar Colors:**
- 🟢 **Green** - Low risk percentage (safe)
- 🟡 **Yellow** - Medium risk percentage (monitor)
- 🔴 **Red** - High risk percentage (address immediately)

---

## Recommended Drills

| Value | Description |
|-------|-------------|
| **Priority Number** | 1 = most important to address first, based on which risk is highest. |
| **Drill Name** | Specific exercise to improve that issue. |
| **Duration** | Suggested practice time for the drill. |
| **Description** | Step-by-step instructions for performing the drill. |

**Drill Assignments:**
| Issue | Recommended Drill |
|-------|-------------------|
| Poor Kinetic Chain | Hip Drive Drill - Practice rotating hips before arm swing |
| Shoulder Overuse | Low Contact Drill - Keep paddle below shoulder height |
| Knee Stress | Athletic Stance Practice - Maintain soft knees at 130-150° |

---

## Recent Analyses

| Value | Description |
|-------|-------------|
| **Stroke Icon** | What stroke type was analyzed. |
| **Stroke Name** | Human-readable name (Serve, Dink, Drive, Overhead). |
| **Date** | When the analysis was completed (Today, Yesterday, Xd ago, or Month Day). |
| **Score %** | Skill score for that specific session. |

---

## Data Sources

All values are calculated from:
1. **`analysis_jobs` table** - Stored in Supabase database
2. **`result_json` field** - Contains frame-by-frame biomechanics data
3. **`result_json.summary`** - Contains aggregated metrics per session
4. **`result_json.frames[].metrics`** - Contains per-frame joint angles

---

## Calculation Formulas

```
Skill Score = 100 - ((shoulder_overuse + poor_kinetic_chain + knee_stress) / 3) × 2

Risk Breakdown = (frames with issue / total analyzed frames) × 100

Trend = latest_session_score - oldest_session_score_this_week
```
