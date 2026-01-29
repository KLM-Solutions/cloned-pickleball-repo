import React from 'react';
import { LucideIcon, ArrowRight, Video, TrendingUp, History } from 'lucide-react';

interface EmptyStateProps {
    title: string;
    description: string;
    actionLabel: string;
    onAction: () => void;
    icon?: LucideIcon;
    variant?: 'default' | 'history' | 'analytics';
}

export function EmptyState({
    title,
    description,
    actionLabel,
    onAction,
    icon: Icon,
    variant = 'default'
}: EmptyStateProps) {

    // Default icons based on variant if not provided
    const DisplayIcon = Icon || (
        variant === 'history' ? History :
            variant === 'analytics' ? TrendingUp :
                Video
    );

    return (
        <div className="flex flex-col items-center justify-center p-8 md:p-12 text-center bg-neutral-50 border border-neutral-200 rounded-2xl animate-in fade-in zoom-in-95 duration-500">
            <div className="w-16 h-16 md:w-20 md:h-20 bg-white rounded-full flex items-center justify-center shadow-sm mb-6 border border-neutral-100">
                <DisplayIcon className="w-8 h-8 md:w-10 md:h-10 text-neutral-400" strokeWidth={1.5} />
            </div>

            <h3 className="text-xl md:text-2xl font-bold text-neutral-900 mb-2">
                {title}
            </h3>

            <p className="text-neutral-500 max-w-md mx-auto mb-8 text-sm md:text-base leading-relaxed">
                {description}
            </p>

            <button
                onClick={onAction}
                className="group relative inline-flex items-center justify-center gap-2 px-6 py-3 text-sm font-semibold text-white bg-black rounded-full hover:bg-neutral-800 transition-all duration-300 shadow-md hover:shadow-lg hover:-translate-y-0.5"
            >
                {actionLabel}
                <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
            </button>
        </div>
    );
}
