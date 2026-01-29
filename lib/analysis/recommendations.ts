/**
 * Comprehensive Coaching Recommendation Templates
 * 
 * 10 Key Biomechanics Parameters with:
 * - Specific action (what to change)
 * - Drill suggestion (how to practice)
 * - Benefit explanation (why it matters)
 * - Visual cue hint (optional reference)
 * 
 * Written in conversational, encouraging tone
 */

import { StrokeType } from './types';

// Recommendation template structure
export interface RecommendationTemplate {
    issue: string;
    title: string;
    emoji: string;
    action: string;
    drill: string;
    benefit: string;
    visualCue?: string;
    strokeVariations?: Partial<Record<StrokeType, { action: string; drill: string }>>;
}

// Full recommendation with formatted output
export interface CoachingRecommendation {
    issue: string;
    title: string;
    severity: 'high' | 'medium' | 'low';
    message: string;        // Short inline message
    fullCoaching: string;   // Rich markdown coaching advice
    drill: string;
    benefit: string;
}

/**
 * Master template library for 10 key biomechanics parameters
 */
const RECOMMENDATION_TEMPLATES: Record<string, RecommendationTemplate> = {
    // ============================================
    // 1. SHOULDER ROTATION
    // ============================================
    shoulder_over_rotation: {
        issue: 'shoulder_over_rotation',
        title: 'Shoulder Over-Rotation',
        emoji: '⚠️',
        action: 'Lower your swing plane and keep your shoulder relaxed. Aim to keep your elbow below shoulder height during the backswing.',
        drill: '**Wall Paddle Drill**: Stand sideways to a wall (about 1 foot away). Practice your swing without touching the wall. This teaches a compact, controlled swing plane. Do 20 reps each side.',
        benefit: 'Reducing shoulder lift decreases rotator cuff strain and improves shot consistency. You\'ll feel less fatigue in longer matches!',
        visualCue: 'Imagine your elbow tracing a line at chest height, never going above your ear.',
        strokeVariations: {
            serve: {
                action: 'Keep your tossing arm controlled. Your hitting shoulder should stay below 140° until contact.',
                drill: '**Trophy Pose Check**: Pause at your highest backswing point. Your elbow should be at ear height, not above your head.'
            },
            overhead: {
                action: 'For overheads, some shoulder elevation is needed. Focus on quick, explosive contact rather than holding the high position.',
                drill: '**Quick Strike Drill**: Have a partner toss balls high. Focus on fast contact, not holding the position.'
            }
        }
    },

    shoulder_under_rotation: {
        issue: 'shoulder_under_rotation',
        title: 'Shoulder Under-Rotation',
        emoji: '💡',
        action: 'Increase your backswing to load more power. Your shoulder should turn to at least 90° on groundstrokes.',
        drill: '**Coil & Freeze Drill**: Turn your shoulders until your back faces the net, then freeze. Feel this coil before every shot. Practice 15 times.',
        benefit: 'A fuller shoulder turn stores more elastic energy, giving you effortless power without straining your arm.',
        visualCue: 'Try to show your back to the net during preparation.'
    },

    shoulder_overuse: {
        issue: 'shoulder_overuse',
        title: 'Shoulder Strain Risk',
        emoji: '🔴',
        action: 'You\'re relying too much on your shoulder for power. Let your hips and core do the work instead.',
        drill: '**Hip-First Drill**: Hold a ball between your elbow and ribs. Swing without dropping it. This forces you to rotate your body, not just your arm. 20 reps.',
        benefit: 'Hip-led swings reduce shoulder stress by 40% and add more pace to your shots. Your shoulder will thank you after long sessions!',
        visualCue: 'Think: Hips rotate → shoulders follow → arm whips through.',
        strokeVariations: {
            serve: {
                action: 'Use your legs and core to drive the serve. Your arm should feel like a whip, not a hammer.',
                drill: '**Knee Bend Serve**: Exaggerate bending your knees before serving. Feel the power come from your legs.'
            }
        }
    },

    // ============================================
    // 2. HIP ROTATION
    // ============================================
    insufficient_hip_rotation: {
        issue: 'insufficient_hip_rotation',
        title: 'Late Hip Turn',
        emoji: '🔄',
        action: 'Your hips are rotating late (or not enough). Initiate hip rotation BEFORE your arm starts forward.',
        drill: '**No-Arm Swings**: Cross your arms over your chest and practice rotating your hips toward the target 20 times. Feel your core engage. Then add the arm motion.',
        benefit: 'Hip rotation adds 30% more power while reducing arm strain. It\'s the secret to effortless pace!',
        visualCue: 'Your belt buckle should point at the target before contact.',
        strokeVariations: {
            groundstroke: {
                action: 'On drives, your hips should lead by rotating at least 45° toward the target.',
                drill: '**Shadow Swings**: Focus on rotating hips before arm swing. Practice 30 reps without a ball.'
            },
            serve: {
                action: 'Your hip should turn and face the net before your arm reaches full extension.',
                drill: '**Hip Bump Drill**: Start in serving stance, bump your front hip toward the net, then let your arm follow.'
            }
        }
    },

    poor_kinetic_chain: {
        issue: 'poor_kinetic_chain',
        title: 'Disconnected Swing',
        emoji: '⛓️',
        action: 'Your power isn\'t flowing efficiently from ground to paddle. Focus on sequential body rotation.',
        drill: '**Chain Reaction Drill**: Practice in slow motion: Push from back foot → rotate hips → rotate shoulders → swing arm. Do 10 slow-motion reps, then 10 at full speed.',
        benefit: 'A connected kinetic chain multiplies your power while distributing effort across your whole body. Less fatigue, more pace!',
        visualCue: 'Think of cracking a whip: The handle (legs) starts, energy flows to the tip (paddle).'
    },

    // ============================================
    // 3. KNEE FLEXION
    // ============================================
    excessive_knee_bend: {
        issue: 'excessive_knee_bend',
        title: 'Too Deep Knee Bend',
        emoji: '🦵',
        action: 'You\'re squatting too low, which limits mobility and stresses your knees. Maintain an athletic stance.',
        drill: '**Mirror Check Drill**: Stand in front of a mirror. Bend until your eyes drop about 6 inches, no more. This is your athletic base. Hold for 30 seconds, rest, repeat 5 times.',
        benefit: 'Proper knee bend improves reaction time and reduces patellar stress. You\'ll move faster to the next shot!',
        visualCue: 'Your thighs should never go below parallel to the ground during play.'
    },

    insufficient_knee_bend: {
        issue: 'insufficient_knee_bend',
        title: 'Standing Too Tall',
        emoji: '📏',
        action: 'Bend your knees more to lower your center of gravity. This improves stability and reach.',
        drill: '**Ready Position Holds**: Get into ready position with knees bent (about 120-140°). Hold for 60 seconds while bouncing slightly. Repeat 3 times between points.',
        benefit: 'Lower stance improves balance, reaction time, and lets you explode to the ball faster.',
        visualCue: 'Imagine sitting on a high bar stool - not standing, not fully sitting.'
    },

    knee_stress: {
        issue: 'knee_stress',
        title: 'Knee Stress Detected',
        emoji: '⚕️',
        action: 'Your knee angle suggests strain. Avoid locking out or excessive bending. Move with soft, springy knees.',
        drill: '**Soft Landing Drill**: Do 10 small hops, focusing on landing with bent knees that absorb impact. Then apply this feeling to your split step.',
        benefit: 'Protecting your knees now prevents long-term injury. Soft, athletic movement is key.',
        visualCue: 'Think "shock absorbers" - knees should never be rigid.'
    },

    // ============================================
    // 4. ELBOW POSITION
    // ============================================
    elbow_form: {
        issue: 'elbow_form',
        title: 'Elbow Position',
        emoji: '💪',
        action: 'Adjust your elbow angle for better paddle control. Keep it consistent through contact.',
        drill: '**Contact Point Drill**: Have a partner hold a ball at your ideal contact point. Practice bringing your paddle to touch it with proper elbow angle. Repeat 20 times.',
        benefit: 'Consistent elbow position gives you reliable contact and better control on every shot.',
        visualCue: 'Your elbow should form a relaxed "L" shape at contact for most shots.',
        strokeVariations: {
            dink: {
                action: 'Keep your elbow close to your body with a 90-110° angle. This gives you fine touch control.',
                drill: '**Kitchen Dinks**: Practice dinks with your elbow pinned to your ribs. Feel the control difference.'
            },
            serve: {
                action: 'Your elbow should be bent at 90° at trophy position, then extend through contact.',
                drill: '**Throw Practice**: Throw a ball overhand. Notice your elbow bends then extends. Mimic this in your serve.'
            }
        }
    },

    elbow_strain: {
        issue: 'elbow_strain',
        title: 'Elbow Strain Risk',
        emoji: '🩹',
        action: 'Your elbow is working too hard. Soften your grip and use more body rotation.',
        drill: '**Loose Grip Hits**: Rate your grip pressure 1-10. Practice hitting at a 4-5 (relaxed). Notice how your arm feels better.',
        benefit: 'A relaxed arm prevents tennis elbow and actually improves your touch. Tension is the enemy!',
        visualCue: 'Grip the paddle like you\'re holding a bird - firm enough it can\'t fly away, gentle enough not to hurt it.'
    },

    // ============================================
    // 5. CONTACT POINT
    // ============================================
    early_contact: {
        issue: 'early_contact',
        title: 'Early Contact',
        emoji: '⏰',
        action: 'You\'re hitting the ball too far in front. Let it come deeper into your stance for better control.',
        drill: '**Delay Drill**: Have a partner feed balls. Count "1, 2" before swinging. This builds patience and timing.',
        benefit: 'Proper contact point improves accuracy and gives you more shot options.',
        visualCue: 'Contact should happen beside your front hip, not way out in front.'
    },

    late_contact: {
        issue: 'late_contact',
        title: 'Late Contact',
        emoji: '⏱️',
        action: 'You\'re contacting the ball too late (beside or behind your body). Prepare earlier and meet it in front.',
        drill: '**Early Prep Drill**: As soon as the ball crosses the net, have your paddle back. Practice 20 balls focusing only on early preparation.',
        benefit: 'Getting to the ball earlier gives you more power and control options.',
        visualCue: 'Your paddle should be ready before the ball bounces on your side.'
    },

    // ============================================
    // 6. WEIGHT TRANSFER
    // ============================================
    poor_weight_transfer: {
        issue: 'poor_weight_transfer',
        title: 'Static Weight',
        emoji: '⚖️',
        action: 'You\'re hitting flat-footed. Shift your weight forward into the shot.',
        drill: '**Step-In Drill**: On every groundstroke, take a small step forward as you hit. Feel your weight moving through the ball. 30 reps.',
        benefit: 'Weight transfer multiplies your power effortlessly. Pro players get 70% of their power from weight shift!',
        visualCue: 'End with 80% of your weight on your front foot after power shots.'
    },

    // ============================================
    // 7. FOLLOW THROUGH
    // ============================================
    incomplete_follow_through: {
        issue: 'incomplete_follow_through',
        title: 'Short Follow-Through',
        emoji: '🎯',
        action: 'Complete your swing! Your paddle should finish over your opposite shoulder.',
        drill: '**Catch Drill**: After hitting, catch your paddle with your non-hitting hand over your opposite shoulder. This ensures full follow-through.',
        benefit: 'Full follow-through adds topspin, pace, and protects your arm by decelerating naturally.',
        visualCue: 'Point your paddle at where you want the ball to go, then wrap it over your shoulder.'
    },

    // ============================================
    // 8. READY POSITION
    // ============================================
    poor_ready_position: {
        issue: 'poor_ready_position',
        title: 'Ready Position',
        emoji: '🏋️',
        action: 'Get into athletic ready position between shots: knees bent, paddle up, weight on balls of feet.',
        drill: '**Split Step Drill**: After each shot, hop and land in ready position. Make this automatic. Practice during warmup.',
        benefit: 'Good ready position shaves precious milliseconds off your reaction time. It\'s the foundation of quick hands!',
        visualCue: 'Paddle at 12 o\'clock, elbows bent, bouncy like a goalkeeper.'
    },

    // ============================================
    // 9. WRIST POSITION
    // ============================================
    wrist_break: {
        issue: 'wrist_break',
        title: 'Excessive Wrist Break',
        emoji: '✋',
        action: 'Your wrist is bending too much. Keep it firm and stable through contact.',
        drill: '**Firm Wrist Volleys**: Practice volleys with your wrist locked. The paddle should feel like an extension of your forearm.',
        benefit: 'A stable wrist gives you consistent contact and reduces strain.',
        visualCue: 'Imagine your wrist is in a soft cast during contact - it can move a little, but stays mostly stable.'
    },

    wrist_locked: {
        issue: 'wrist_locked',
        title: 'Wrist Too Stiff',
        emoji: '🤖',
        action: 'Relax your wrist slightly. Some flexibility adds touch and spin potential.',
        drill: '**Wrist Flicks**: Practice dinks using only wrist motion. Exaggerate the looseness. Then integrate this feeling into normal shots.',
        benefit: 'A relaxed wrist improves touch on soft shots and reduces arm tension.',
        visualCue: 'Your wrist should be like a door hinge - can move but has control.'
    },

    // ============================================
    // 10. BALANCE & STABILITY
    // ============================================
    off_balance: {
        issue: 'off_balance',
        title: 'Balance Issue',
        emoji: '⚡',
        action: 'You\'re reaching too far or not recovering properly. Stay centered and move your feet.',
        drill: '**Freeze Frame**: After each practice shot, freeze and check if you could push off in any direction. If not, you reached. Move your feet instead.',
        benefit: 'Better balance means you\'re always ready for the next shot. Champions recover to ready position instantly.',
        visualCue: 'Your head should stay level during shots - if it dips, you\'re off balance.'
    },

    poor_recovery: {
        issue: 'poor_recovery',
        title: 'Slow Recovery',
        emoji: '🔙',
        action: 'Return to ready position faster after each shot. Recovery is as important as the shot itself.',
        drill: '**Recovery Ladder**: Hit a shot, shuffle back to center, touch the ground, get ready. Repeat 20 times.',
        benefit: 'Quick recovery positions you for the next ball and pressures your opponent.',
        visualCue: 'Finish every shot moving TOWARD ready position, not falling away from it.'
    },
};

/**
 * Get enhanced recommendation for an issue type
 */
export function getEnhancedRecommendation(
    issueType: string,
    strokeType?: StrokeType | string,
    severity: 'high' | 'medium' | 'low' = 'medium'
): CoachingRecommendation {
    const template = RECOMMENDATION_TEMPLATES[issueType];

    if (!template) {
        return {
            issue: issueType,
            title: formatIssueTitle(issueType),
            severity,
            message: 'Work on this aspect of your technique.',
            fullCoaching: 'Focus on consistent practice to improve this area.',
            drill: 'Practice with deliberate focus on this movement.',
            benefit: 'Improvement here will enhance your overall game.'
        };
    }

    // Check for stroke-specific variations
    const stroke = (strokeType as StrokeType) || 'groundstroke';
    const variation = template.strokeVariations?.[stroke];

    const action = variation?.action || template.action;
    const drill = variation?.drill || template.drill;

    // Build full coaching message
    const fullCoaching = `${template.emoji} **${template.title}**

${action}

**Try this drill:**
${drill}

**Why it matters:**
${template.benefit}${template.visualCue ? `

💡 *Visual cue: ${template.visualCue}*` : ''}`;

    return {
        issue: issueType,
        title: template.title,
        severity,
        message: action,
        fullCoaching,
        drill: drill.replace(/\*\*/g, ''), // Strip markdown for plain text
        benefit: template.benefit
    };
}

/**
 * Get all recommendations for a list of issues
 */
export function getRecommendationsForIssues(
    issues: { type: string; severity?: 'high' | 'medium' | 'low' }[],
    strokeType?: StrokeType | string
): CoachingRecommendation[] {
    return issues.map(issue =>
        getEnhancedRecommendation(issue.type, strokeType, issue.severity || 'medium')
    );
}

/**
 * Format issue type to human-readable title
 */
function formatIssueTitle(issueType: string): string {
    return issueType
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

/**
 * Get all available issue types
 */
export function getAllIssueTypes(): string[] {
    return Object.keys(RECOMMENDATION_TEMPLATES);
}

/**
 * Get quick recommendation text (for inline display)
 */
export function getQuickRecommendation(issueType: string): string {
    const template = RECOMMENDATION_TEMPLATES[issueType];
    if (!template) {
        return 'Work on this aspect of your technique.';
    }
    return template.action;
}

/**
 * Get drill for an issue type
 */
export function getDrillForIssue(issueType: string, strokeType?: StrokeType | string): string {
    const template = RECOMMENDATION_TEMPLATES[issueType];
    if (!template) {
        return 'Practice with deliberate focus on this movement.';
    }

    const stroke = (strokeType as StrokeType) || 'groundstroke';
    return template.strokeVariations?.[stroke]?.drill || template.drill;
}
