/**
 * Demo Analysis Data
 * Real analysis data from a completed session - showcases injury risks and performance insights
 */

export const DEMO_VIDEO_URL = "https://tnfqqcjstysyuqajfoqr.supabase.co/storage/v1/object/public/analysis-results/762644e0-0192-49d9-bf85-a127c3b56449/annotated.mp4";

export const DEMO_ANALYSIS_DATA = {
    videoUrl: DEMO_VIDEO_URL,
    stroke_type: "serve",
    summary: {
        total_frames: 439,
        duration_sec: 14.6,
        fps: 30,
        overall_score: 68,
        injury_risk_summary: "Medium - Poor kinetic chain and elbow strain detected",
    },
    frames: [
        {
            frameIdx: 0,
            timestampSec: 0,
            track_id: 0,
            confidence: 0.919427752494812,
            injury_risk: "medium",
            injury_risks: [
                {
                    type: "poor_kinetic_chain",
                    angle: 5.9,
                    message: "Insufficient hip rotation (5.9°)",
                    severity: "medium",
                    recommendation: "Engage hips and core for power. Reduce arm strain."
                }
            ],
            feedback: ["FAULT: Contact point too high (Above waist)", "Power: Rotate hips more before contact"],
            metrics: {
                nose_y: 0.2255,
                right_hip_y: 0.622,
                right_wrist_y: 0.5462,
                hip_rotation_deg: 5.9,
                wrist_above_head: false,
                left_knee_flexion: 167.6,
                wrist_above_waist: true,
                left_elbow_flexion: 144.8,
                right_knee_flexion: 159.2,
                right_elbow_flexion: 129,
                left_shoulder_abduction: 18.1,
                right_shoulder_abduction: 38.6
            }
        },
        {
            frameIdx: 1,
            timestampSec: 0.033,
            track_id: 0,
            confidence: 0.9195222854614258,
            injury_risk: "medium",
            injury_risks: [
                {
                    type: "poor_kinetic_chain",
                    angle: 6.3,
                    message: "Insufficient hip rotation (6.3°)",
                    severity: "medium",
                    recommendation: "Engage hips and core for power. Reduce arm strain."
                },
                {
                    type: "elbow_strain",
                    angle: 142.5,
                    message: "Elbow angle (142.5°) outside optimal range for serve",
                    severity: "low",
                    recommendation: "Optimal elbow angle for serve: 90-130°"
                }
            ],
            feedback: ["FAULT: Contact point too high (Above waist)", "Power: Rotate hips more before contact", "Elbow angle (142.5°) - optimal is 90-130° for serve"],
            metrics: {
                nose_y: 0.2269,
                right_hip_y: 0.625,
                right_wrist_y: 0.5356,
                hip_rotation_deg: 6.3,
                wrist_above_head: false,
                left_knee_flexion: 169.8,
                wrist_above_waist: true,
                left_elbow_flexion: 147.5,
                right_knee_flexion: 160.6,
                right_elbow_flexion: 142.5,
                left_shoulder_abduction: 38,
                right_shoulder_abduction: 37.1
            }
        },
        {
            frameIdx: 2,
            timestampSec: 0.067,
            track_id: 0,
            confidence: 0.9228397607803345,
            injury_risk: "medium",
            injury_risks: [
                {
                    type: "poor_kinetic_chain",
                    angle: 9.3,
                    message: "Insufficient hip rotation (9.3°)",
                    severity: "medium",
                    recommendation: "Engage hips and core for power. Reduce arm strain."
                },
                {
                    type: "elbow_strain",
                    angle: 133.9,
                    message: "Elbow angle (133.9°) outside optimal range for serve",
                    severity: "low",
                    recommendation: "Optimal elbow angle for serve: 90-130°"
                }
            ],
            feedback: ["FAULT: Contact point too high (Above waist)", "Power: Rotate hips more before contact", "Elbow angle (133.9°) - optimal is 90-130° for serve"],
            metrics: {
                nose_y: 0.2357,
                right_hip_y: 0.6317,
                right_wrist_y: 0.5157,
                hip_rotation_deg: 9.3,
                wrist_above_head: false,
                left_knee_flexion: 148.6,
                wrist_above_waist: true,
                left_elbow_flexion: 136.6,
                right_knee_flexion: 144,
                right_elbow_flexion: 133.9,
                left_shoulder_abduction: 24,
                right_shoulder_abduction: 37.6
            }
        },
        {
            frameIdx: 3,
            timestampSec: 0.1,
            track_id: 0,
            confidence: 0.8034681677818298,
            injury_risk: "medium",
            injury_risks: [
                {
                    type: "poor_kinetic_chain",
                    angle: 11.2,
                    message: "Insufficient hip rotation (11.2°)",
                    severity: "medium",
                    recommendation: "Engage hips and core for power. Reduce arm strain."
                },
                {
                    type: "elbow_strain",
                    angle: 145.1,
                    message: "Elbow angle (145.1°) outside optimal range for serve",
                    severity: "low",
                    recommendation: "Optimal elbow angle for serve: 90-130°"
                }
            ],
            feedback: ["FAULT: Contact point too high (Above waist)", "Elbow angle (145.1°) - optimal is 90-130° for serve"],
            metrics: {
                nose_y: 0.2499,
                right_hip_y: 0.6438,
                right_wrist_y: 0.5577,
                hip_rotation_deg: 11.2,
                wrist_above_head: false,
                left_knee_flexion: 166.6,
                wrist_above_waist: true,
                left_elbow_flexion: 139.6,
                right_knee_flexion: 158.4,
                right_elbow_flexion: 145.1,
                left_shoulder_abduction: 23.1,
                right_shoulder_abduction: 31.6
            }
        },
        {
            frameIdx: 4,
            timestampSec: 0.133,
            track_id: 0,
            confidence: 0.806236982345581,
            injury_risk: "medium",
            injury_risks: [
                {
                    type: "poor_kinetic_chain",
                    angle: 6.1,
                    message: "Insufficient hip rotation (6.1°)",
                    severity: "medium",
                    recommendation: "Engage hips and core for power. Reduce arm strain."
                }
            ],
            feedback: ["FAULT: Contact point too high (Above waist)", "Power: Rotate hips more before contact"],
            metrics: {
                nose_y: 0.3395,
                right_hip_y: 0.6575,
                right_wrist_y: 0.5793,
                hip_rotation_deg: 6.1,
                wrist_above_head: false,
                left_knee_flexion: 163.8,
                wrist_above_waist: true,
                left_elbow_flexion: 151.1,
                right_knee_flexion: 152.2,
                right_elbow_flexion: 125.1,
                left_shoulder_abduction: 18.1,
                right_shoulder_abduction: 45
            }
        },
        {
            frameIdx: 5,
            timestampSec: 0.167,
            track_id: 0,
            confidence: 0.7737314701080322,
            injury_risk: "medium",
            injury_risks: [
                {
                    type: "poor_kinetic_chain",
                    angle: 6.5,
                    message: "Insufficient hip rotation (6.5°)",
                    severity: "medium",
                    recommendation: "Engage hips and core for power. Reduce arm strain."
                },
                {
                    type: "elbow_strain",
                    angle: 158.3,
                    message: "Elbow angle (158.3°) outside optimal range for serve",
                    severity: "low",
                    recommendation: "Optimal elbow angle for serve: 90-130°"
                }
            ],
            feedback: ["FAULT: Contact point too high (Above waist)", "Power: Rotate hips more before contact", "Elbow angle (158.3°) - optimal is 90-130° for serve"],
            metrics: {
                nose_y: 0.196,
                right_hip_y: 0.5848,
                right_wrist_y: 0.5795,
                hip_rotation_deg: 6.5,
                wrist_above_head: false,
                left_knee_flexion: 133.9,
                wrist_above_waist: true,
                left_elbow_flexion: 153.2,
                right_knee_flexion: 162.7,
                right_elbow_flexion: 158.3,
                left_shoulder_abduction: 8.7,
                right_shoulder_abduction: 46.7
            }
        },
        {
            frameIdx: 6,
            timestampSec: 0.2,
            track_id: 0,
            confidence: 0.8389717936515808,
            injury_risk: "medium",
            injury_risks: [
                {
                    type: "poor_kinetic_chain",
                    angle: 0,
                    message: "Insufficient hip rotation (0°)",
                    severity: "medium",
                    recommendation: "Engage hips and core for power. Reduce arm strain."
                },
                {
                    type: "elbow_strain",
                    angle: 0,
                    message: "Elbow angle (0°) outside optimal range for serve",
                    severity: "low",
                    recommendation: "Optimal elbow angle for serve: 90-130°"
                }
            ],
            feedback: [],
            metrics: {
                nose_y: null,
                right_hip_y: null,
                right_wrist_y: null,
                hip_rotation_deg: null,
                wrist_above_head: false,
                left_knee_flexion: null,
                wrist_above_waist: false,
                left_elbow_flexion: null,
                right_knee_flexion: null,
                right_elbow_flexion: null,
                left_shoulder_abduction: null,
                right_shoulder_abduction: null
            }
        },
        {
            frameIdx: 7,
            timestampSec: 0.233,
            track_id: 0,
            confidence: 0.8365092873573303,
            injury_risk: "medium",
            injury_risks: [
                {
                    type: "poor_kinetic_chain",
                    angle: 0,
                    message: "Insufficient hip rotation (0°)",
                    severity: "medium",
                    recommendation: "Engage hips and core for power. Reduce arm strain."
                },
                {
                    type: "elbow_strain",
                    angle: 0,
                    message: "Elbow angle (0°) outside optimal range for serve",
                    severity: "low",
                    recommendation: "Optimal elbow angle for serve: 90-130°"
                }
            ],
            feedback: [],
            metrics: {
                nose_y: null,
                right_hip_y: null,
                right_wrist_y: null,
                hip_rotation_deg: null,
                wrist_above_head: false,
                left_knee_flexion: null,
                wrist_above_waist: false,
                left_elbow_flexion: null,
                right_knee_flexion: null,
                right_elbow_flexion: null,
                left_shoulder_abduction: null,
                right_shoulder_abduction: null
            }
        },
        {
            frameIdx: 8,
            timestampSec: 0.267,
            track_id: 0,
            confidence: 0.8312113285064697,
            injury_risk: "medium",
            injury_risks: [
                {
                    type: "poor_kinetic_chain",
                    angle: 0,
                    message: "Insufficient hip rotation (0°)",
                    severity: "medium",
                    recommendation: "Engage hips and core for power. Reduce arm strain."
                },
                {
                    type: "elbow_strain",
                    angle: 0,
                    message: "Elbow angle (0°) outside optimal range for serve",
                    severity: "low",
                    recommendation: "Optimal elbow angle for serve: 90-130°"
                }
            ],
            feedback: [],
            metrics: {
                nose_y: null,
                right_hip_y: null,
                right_wrist_y: null,
                hip_rotation_deg: null,
                wrist_above_head: false,
                left_knee_flexion: null,
                wrist_above_waist: false,
                left_elbow_flexion: null,
                right_knee_flexion: null,
                right_elbow_flexion: null,
                left_shoulder_abduction: null,
                right_shoulder_abduction: null
            }
        },
        {
            frameIdx: 9,
            timestampSec: 0.3,
            track_id: 0,
            confidence: 0.8283801078796387,
            injury_risk: "medium",
            injury_risks: [
                {
                    type: "poor_kinetic_chain",
                    angle: 0,
                    message: "Insufficient hip rotation (0°)",
                    severity: "medium",
                    recommendation: "Engage hips and core for power. Reduce arm strain."
                },
                {
                    type: "elbow_strain",
                    angle: 0,
                    message: "Elbow angle (0°) outside optimal range for serve",
                    severity: "low",
                    recommendation: "Optimal elbow angle for serve: 90-130°"
                }
            ],
            feedback: [],
            metrics: {
                nose_y: null,
                right_hip_y: null,
                right_wrist_y: null,
                hip_rotation_deg: null,
                wrist_above_head: false,
                left_knee_flexion: null,
                wrist_above_waist: false,
                left_elbow_flexion: null,
                right_knee_flexion: null,
                right_elbow_flexion: null,
                left_shoulder_abduction: null,
                right_shoulder_abduction: null
            }
        },
        {
            frameIdx: 10,
            timestampSec: 0.333,
            track_id: 0,
            confidence: 0.6176885366439819,
            injury_risk: "medium",
            injury_risks: [
                {
                    type: "poor_kinetic_chain",
                    angle: 0,
                    message: "Insufficient hip rotation (0°)",
                    severity: "medium",
                    recommendation: "Engage hips and core for power. Reduce arm strain."
                }
            ],
            feedback: [],
            metrics: {
                nose_y: null,
                right_hip_y: null,
                right_wrist_y: null,
                hip_rotation_deg: null,
                wrist_above_head: false,
                left_knee_flexion: null,
                wrist_above_waist: false,
                left_elbow_flexion: null,
                right_knee_flexion: null,
                right_elbow_flexion: null,
                left_shoulder_abduction: null,
                right_shoulder_abduction: null
            }
        }
    ],
    insights: [
        {
            type: "injury_risk",
            severity: "medium",
            title: "Poor Kinetic Chain",
            description: "Hip rotation is insufficient throughout the serve, ranging from 5.9° to 11.2°.",
            recommendation: "Engage hips and core for power generation. Reduce arm strain by using the kinetic chain.",
        },
        {
            type: "injury_risk",
            severity: "low",
            title: "Elbow Strain Risk",
            description: "Elbow angle (142.5° - 158.3°) exceeds the optimal range of 90-130° for serves.",
            recommendation: "Keep elbow angle within optimal range to reduce strain risk.",
        },
        {
            type: "performance",
            severity: "warning",
            title: "Contact Point Too High",
            description: "Ball contact is occurring above waist level, which may result in a fault.",
            recommendation: "Lower your contact point to ensure legal serve.",
        },
    ],
    llmResponse: `## Overall Assessment
Your serve technique exhibits several critical areas for improvement, particularly concerning the kinetic chain and elbow positioning. Addressing these issues will not only enhance your serve power and accuracy but also reduce the risk of injury.

## Key Strengths
- **Consistent Ball Toss**: Your toss remains fairly controlled, allowing for a reliable starting point in your serve.
- **Engagement of Upper Body**: You demonstrate a willingness to use your shoulders and arms, which can be an advantage once your mechanics are improved.

## Priority Areas for Improvement
1. **Kinetic Chain Coordination**:
   - **Issue**: Poor Kinetic Chain detected 15 times throughout the serve.
   - **Action**: Focus on using your hips and legs more effectively to generate power. Your hip angles are often too low (e.g., 0-6° at various timestamps).

2. **Elbow Positioning**:
   - **Issue**: Elbow Strain Risk also noted 15 times, with angles indicating excessive strain (e.g., 172° at 4.0s and 162° at 3.0s).
   - **Action**: Work on keeping the elbow stable and reducing the angle to below 150° during the serve. This will help in distributing the force more evenly through your arm.

3. **Shoulder Rotation**:
   - **Issue**: Shoulder angles are often too high (e.g., 103° at 8.0s and 114° at 10.0s).
   - **Action**: Aim to maintain a shoulder angle under 90° during the serve preparation and execution to enhance serve mechanics and reduce strain.

4. **Knee Stability**:
   - **Issue**: Knee Stress noted at 12.1s with a 83° angle.
   - **Action**: Ensure that your knees are aligned and stable throughout the serve, focusing on maintaining a slight bend and using them for balance.

## Drill Recommendations
1. **Hip Engagement Drill**:
   - Stand with your feet shoulder-width apart. Practice hip rotation by mimicking the serve motion without the ball. Focus on rotating your hips while keeping your shoulders aligned. Gradually increase the speed and intensity, ensuring your hips lead the motion.

2. **Elbow Stability Drill**:
   - Use a resistance band tied to a stationary object. Grip the band and perform a mock serve while keeping your elbow close to your body. This will help reinforce proper elbow positioning and strength without adding strain.

3. **Knee Alignment Drill**:
   - Perform wall sits to strengthen your quadriceps and ensure proper knee alignment. Hold the position for 30 seconds, focusing on keeping your knees behind your toes. This will enhance your stability during the serve.

## Injury Prevention Notes
- **Elbow Strain Risk**: Given the repeated strain risk, it's crucial to incorporate stretching and strengthening exercises for the forearm and shoulder. Focus on exercises that promote flexibility and stability in these areas.
- **Knee Stress**: Pay attention to your knee alignment and avoid overextending. Stronger quads and hamstrings will provide better support during dynamic movements.

By systematically addressing these areas, you'll not only improve your serve but also establish a strong foundation for your overall performance in pickleball. Keep practicing, and remember to listen to your body as you implement these changes!`
};
