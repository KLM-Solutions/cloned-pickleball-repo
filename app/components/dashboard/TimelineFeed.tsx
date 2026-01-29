import { TrendingUp } from "lucide-react";

interface Moment {
    time: string;
    type: "success" | "warning" | "danger";
    title: string;
    description: string;
    timestampSec?: number;
}

interface TimelineFeedProps {
    moments: Moment[];
    onMomentClick?: (sec: number) => void;
}

export function TimelineFeed({ moments, onMomentClick }: TimelineFeedProps) {
    return (
        <div className="bg-background rounded-2xl p-4 shadow-sm border border-border">
            <h3 className="text-sm font-bold text-secondary mb-4 flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-success" />
                Performance Timeline
            </h3>
            <div className="space-y-3">
                {moments.map((moment, idx) => (
                    <button
                        key={idx}
                        onClick={() => moment.timestampSec !== undefined && onMomentClick?.(moment.timestampSec)}
                        className="w-full flex items-start gap-3 text-left hover:bg-gray-50 p-2 rounded-lg transition"
                    >
                        <div className={`w-3 h-3 rounded-full mt-1.5 flex-shrink-0 ${moment.type === 'success' ? 'bg-success' :
                                moment.type === 'warning' ? 'bg-accent' : 'bg-red-500'
                            }`} />
                        <div className="flex-1 min-w-0 pb-3 border-b border-gray-100 last:border-0">
                            <div className="flex items-center gap-2 mb-1">
                                <span className="text-xs font-mono text-text-secondary">{moment.time}</span>
                                <span className="text-sm font-semibold text-secondary">{moment.title}</span>
                            </div>
                            <p className="text-xs text-text-secondary">{moment.description}</p>
                        </div>
                    </button>
                ))}
            </div>
        </div>
    );
}
