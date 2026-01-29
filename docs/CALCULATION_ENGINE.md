# StrikeSense Calculation Engine Documentation

This document explains all biomechanical calculations performed by StrikeSense for skill improvement and injury prevention.

---

## Part 1: Joint Angle Calculations

### How Angles Are Calculated

Calculations use a **Hybrid 2D/3D Approach** to maximize accuracy where depth data is reliable:

1.  **Joint Angles (Elbow, Knee, Shoulder)**: Calculated in **3D**.
    *   Uses `x, y, z` coordinates from MediaPipe Pose.
    *   Formula: `cos(θ) = (BA · BC) / (|BA| × |BC|)` (Vector Dot Product)
    *   *Why 3D?* Calculating limb angles strictly in 2D is inaccurate when limbs are foreshortened (pointing at camera).

2.  **Hip Rotation**:
    *   **TypeScript (Frontend)**: Calculated in **3D**. Uses `x, z` (depth) coordinates to measure rotation relative to camera plane (`atan(dz/dx)`).
    *   **Python (Backend)**: Calculated in **2D**. Uses `x, y` projection (`arctan(|Y_diff| / |X_diff|)`).

Where:
- **Points A, B, C** form an angle at point B
- **BA** and **BC** are vectors from B to A and B to C
- The result is converted from radians to degrees

---

### 1. Elbow Flexion

**Points used:** Shoulder → Elbow → Wrist

**What it measures:** How bent or straight the elbow is

| Angle | Meaning |
|-------|---------|
| 180° | Fully straight arm |
| 90° | Right angle at elbow |
| < 90° | Tightly bent |

**Optimal ranges by stroke:**
| Stroke | Min | Max |
|--------|-----|-----|
| Serve | 90° | 130° |
| Dink | 90° | 110° |
| Drive | 120° | 160° |
| Overhead | 90° | 170° |
| Volley | 90° | 120° |

---

### 2. Knee Flexion

**Points used:** Hip → Knee → Ankle

**What it measures:** How bent the knees are during the stance

| Angle | Meaning |
|-------|---------|
| 180° | Standing straight (locked knees) |
| 120-160° | Athletic stance ✅ |
| < 90° | Excessive squat ⚠️ |

**Why it matters:**
- Bent knees provide stability and power
- Too deep (< 90°) = patellar stress risk
- Too straight (> 170°) = reduced mobility

---

### 3. Shoulder Abduction

**Points used:** Hip → Shoulder → Elbow

**What it measures:** How high the arm is raised away from the body

| Angle | Meaning |
|-------|---------|
| 0-40° | Arm at side |
| 40-90° | Normal motion |
| 90-140° | Elevated arm |
| > 140° | Over-elevated ⚠️ |

**CRITICAL THRESHOLD: 140°**
- Raising arm above 140° repeatedly = rotator cuff injury risk ("shoulder impingement")
- Exception: Overhead smashes require higher elevation

---

### 4. Hip Rotation

**Points used:** Left Hip and Right Hip positions

**Formula:**
```
rotation = arctan(|Y difference| / |X difference|) × (180/π)
```

**What it measures:** How much the hips rotate toward the target during the stroke

| Angle | Meaning |
|-------|---------|
| 0-15° | Minimal rotation (arm-only power) |
| 15-30° | Some rotation |
| 30°+ | Good kinetic chain ✅ |
| 45°+ | Excellent rotation |

**Why it matters:**
- Power strokes NEED hip rotation (≥30°)
- Without hip rotation = "Poor Kinetic Chain"
- Forces arm to generate all power = strain risk

---

## Part 2: Injury Risk Detection

### Risk Categories

| Risk Type | Threshold | Severity |
|-----------|-----------|----------|
| **Shoulder Overuse** | > 140° abduction (non-overhead) | Medium/High |
| **Poor Kinetic Chain** | < 30° hip rotation (power strokes) | Medium |
| **Knee Stress** | < 90° flexion (too bent) | Medium |
| **Elbow Strain** | Outside stroke-specific range | Low |

### Overall Risk Level Determination

```typescript
if (shoulder_overuse > 10%) → HIGH
if (poor_kinetic_chain > 20% AND knee_stress > 15%) → HIGH
if (shoulder_overuse > 5% OR poor_kinetic_chain > 15% OR knee_stress > 10%) → MEDIUM
else → LOW
```

---

## Part 3: Session Summary Calculations

### Risk Percentages

For each risk type:
```
percentage = (frames_with_issue / total_frames) × 100
```

Example:
- 100 total frames analyzed
- Shoulder overuse detected in 8 frames
- Percentage = 8%

---

### Skill Score Calculation

The **Skill Score (0-100)** is a composite metric that primarily evaluates **biomechanical efficiency and safety**. It is calculated inversely from your injury risks—meaning perfect form (0% risk) results in a perfect score (100).

#### Formula
$$ \text{Score} = 100 - (2 \times \text{Average Risk \%}) $$

Where **Average Risk %** is the mean of:
1.  **Shoulder Overuse %** (Frames with arm > 140°)
2.  **Poor Kinetic Chain %** (Power strokes with hip rotation < 30°)
3.  **Knee Stress %** (Frames with knee flexion < 90°)

#### Why this formula?
The formula applies a **2x penalty** to risk. This is intentional:
- **Safety First**: Biomechanical flaws are treated as serious detractors. A session with "moderate" risk (e.g., 25% avg risk) results in a score of 50, effectively failing the safety check.
- **High Standards**: To achieve a "Pro" score (>90), you must demonstrate consistent form with very few deviations.

#### Score Interpretation
| Score | Rating | Interpretation |
|-------|--------|----------------|
| **90-100** | Pro | Excellent biomechanics. Minimal injury risk and highly efficient power transfer. |
| **80-89** | Advanced | Solid technique with minor inconsistencies. |
| **60-79** | Intermediate | Noticeable form issues. Focus on specific drills to correct the kinetic chain. |
| **< 60** | Novice / High Risk | Significant biomechanical flaws detected. High risk of injury or inefficiency. |

#### Example Calculation
- **Shoulder Overuse**: 5% of frames
- **Poor Kinetic Chain**: 10% of strokes
- **Knee Stress**: 3% of frames
- **Average Risk** = (5 + 10 + 3) / 3 = **6%**
- **Calculation** = 100 - (6 × 2) = 100 - 12 = **88** (Advanced)

---

### Weekly Trend Calculation

```typescript
// Get sessions with scores from last 7 days
const recentSessions = sessions.filter(s => s.date > 7_days_ago && s.score > 0);

// Compare oldest to newest
if (recentSessions.length >= 2) {
  trend = newestScore - oldestScore;
}
```

- **Positive trend** → ↑ (improving)
- **Negative trend** → ↓ (declining)

---

## Part 4: Stroke-Specific Analysis

Different strokes have different biomechanical requirements:

### Serve
- Requires moderate hip rotation (30°+)
- Shoulder should stay < 140°
- Elbow: 90-130°

### Dink
- Minimal power needed
- Low paddle position
- Soft, controlled movement
- Elbow: 90-110°

### Drive (Groundstroke)
- Maximum hip rotation needed (45°+)
- Weight transfer important
- Elbow: 120-160°

### Overhead
- ONLY stroke where shoulder > 140° is acceptable
- Full arm extension needed
- Elbow: 90-170°

### Volley
- Quick, compact motion
- Firm wrist
- Elbow: 90-120°

---

## Part 5: Drill Recommendations Logic

Based on detected issues, drills are recommended with priority:

```typescript
const DRILL_RECOMMENDATIONS = {
  poor_kinetic_chain: {
    name: "Hip Drive Drill",
    description: "Practice rotating hips before arm swing",
    duration: "5 mins",
  },
  shoulder_overuse: {
    name: "Low Contact Drill",
    description: "Keep paddle below shoulder height",
    duration: "5 mins",
  },
  knee_stress: {
    name: "Athletic Stance Practice",
    description: "Maintain soft knees at 130-150°",
    duration: "3 mins",
  },
};
```

**Priority Assignment:**
1. Issues with highest percentage get priority 1
2. Lower percentages get lower priority
3. Only issues > 2% are shown

---

## Part 6: Data Flow Summary

```
1. Video Upload
      ↓
2. MediaPipe Pose Detection (33 landmarks per frame)
      ↓
3. Angle Calculations (angles.ts)
   - calculateAngle() for each joint
   - calculateHipRotation()
      ↓
4. Risk Detection (risk.ts)
   - detectRisks() per frame
   - Compare angles to thresholds
      ↓
5. Session Summary (index.ts)
   - calculateRiskPercentages()
   - getOverallRisk()
      ↓
6. Store in Supabase (result_json field)
      ↓
7. Analytics Dashboard (api/analytics/summary)
   - Aggregate across sessions
   - Calculate skill score
   - Generate drill recommendations
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `lib/analysis/angles.ts` | Joint angle calculations from landmarks |
| `lib/analysis/risk.ts` | Injury risk detection with thresholds |
| `lib/analysis/feedback.ts` | Coaching feedback message generation |
| `lib/analysis/index.ts` | Main orchestration of analysis |
| `lib/analysis/types.ts` | TypeScript interfaces |
| `api/analytics/summary/route.ts` | Dashboard data aggregation |

---

## Medical/Research Basis

The thresholds are based on sports medicine research:

- **Shoulder 140° threshold**: Beyond this angle, the supraspinatus tendon can impinge on the acromion (shoulder impingement syndrome)
- **Hip rotation 30° minimum**: Studies show proper kinetic chain reduces arm strain by 40-60%
- **Knee flexion 90° minimum**: Deep flexion beyond this increases patellofemoral joint stress
