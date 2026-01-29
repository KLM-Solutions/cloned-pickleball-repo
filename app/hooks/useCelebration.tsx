import { useCallback, useEffect, useState } from 'react';
import confetti from 'canvas-confetti';

export function useCelebration() {
    const [isEnabled, setIsEnabled] = useState(true);

    // Check system preference for reduced motion
    useEffect(() => {
        const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
        setIsEnabled(!mediaQuery.matches);

        const handler = (e: MediaQueryListEvent) => setIsEnabled(!e.matches);
        mediaQuery.addEventListener('change', handler);
        return () => mediaQuery.removeEventListener('change', handler);
    }, []);

    const triggerConfetti = useCallback(() => {
        if (!isEnabled) return;

        const duration = 3000;
        const animationEnd = Date.now() + duration;
        const defaults = { startVelocity: 30, spread: 360, ticks: 60, zIndex: 100 };

        const randomInRange = (min: number, max: number) => Math.random() * (max - min) + min;

        const interval: any = setInterval(function () {
            const timeLeft = animationEnd - Date.now();

            if (timeLeft <= 0) {
                return clearInterval(interval);
            }

            const particleCount = 50 * (timeLeft / duration);

            // Random confetti bursts
            confetti({
                ...defaults,
                particleCount,
                origin: { x: randomInRange(0.1, 0.3), y: Math.random() - 0.2 }
            });
            confetti({
                ...defaults,
                particleCount,
                origin: { x: randomInRange(0.7, 0.9), y: Math.random() - 0.2 }
            });
        }, 250);
    }, [isEnabled]);

    const triggerBurst = useCallback(() => {
        if (!isEnabled) return;

        confetti({
            particleCount: 100,
            spread: 70,
            origin: { y: 0.6 }
        });
    }, [isEnabled]);

    return { triggerConfetti, triggerBurst, isEnabled };
}
