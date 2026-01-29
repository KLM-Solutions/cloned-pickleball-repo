import React from "react";

interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> {
    className?: string;
}

export function Skeleton({ className, ...props }: SkeletonProps) {
    return (
        <div
            className={`animate-pulse rounded-md bg-neutral-200/80 ${className || ""}`}
            {...props}
        />
    );
}

export function CardSkeleton() {
    return (
        <div className="p-4 border border-neutral-200 rounded-xl space-y-3">
            <Skeleton className="h-4 w-1/3" />
            <Skeleton className="h-20 w-full rounded-lg" />
            <div className="flex justify-between">
                <Skeleton className="h-3 w-1/4" />
                <Skeleton className="h-3 w-1/4" />
            </div>
        </div>
    );
}

export function SingleLineSkeleton() {
    return (
        <div className="flex items-center space-x-4 w-full">
            <Skeleton className="h-10 w-10 rounded-full" />
            <div className="space-y-2 flex-1">
                <Skeleton className="h-3 w-3/4" />
                <Skeleton className="h-3 w-1/2" />
            </div>
        </div>
    );
}
