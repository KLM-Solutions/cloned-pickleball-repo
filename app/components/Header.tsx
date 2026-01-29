"use client";

import { useRouter, usePathname } from "next/navigation";
import { Clock } from "lucide-react";
import {
    SignInButton,
    SignUpButton,
    SignedIn,
    SignedOut,
    UserButton,
} from "@clerk/nextjs";

export default function Header() {
    const router = useRouter();
    const pathname = usePathname();



    return (
        <header className="relative z-10 flex items-center justify-between px-4 md:px-6 py-4 md:py-5 border-b border-neutral-200 bg-white">
            <div className="flex items-center gap-3 md:gap-4">
                <div className="flex items-center gap-2 md:gap-3">
                    <div className="w-9 h-9 md:w-10 md:h-10 rounded-xl bg-black flex items-center justify-center text-lg md:text-xl">
                        🏓
                    </div>
                    <div>
                        <h1 className="text-lg md:text-xl font-bold tracking-tight text-neutral-900">StrikeSense</h1>

                    </div>
                </div>
                <div className="hidden lg:flex items-center gap-2 px-3 py-1.5 rounded-full bg-neutral-100 border border-neutral-200 text-xs text-neutral-600">
                    <span className="w-1.5 h-1.5 rounded-full bg-black animate-pulse" />
                    Coach Mode Active
                </div>
            </div>

            <div className="flex items-center gap-2 md:gap-3">
                {/* Auth Buttons */}
                <SignedOut>
                    <SignInButton mode="modal">
                        <button className="flex items-center gap-1.5 md:gap-2 px-2.5 md:px-3 py-2 text-xs md:text-sm font-medium text-white bg-black rounded-full hover:bg-neutral-800 transition-all duration-300 cursor-pointer">
                            Sign In
                        </button>
                    </SignInButton>
                    <SignUpButton mode="modal">
                        <button className="flex items-center gap-1.5 md:gap-2 px-2.5 md:px-3 py-2 text-xs md:text-sm font-medium text-neutral-500 border border-neutral-200 rounded-full hover:bg-neutral-100 hover:text-black transition-all duration-300 cursor-pointer">
                            Join Team
                        </button>
                    </SignUpButton>
                </SignedOut>
                <SignedIn>
                    <UserButton
                        appearance={{
                            elements: {
                                avatarBox: "w-8 h-8 md:w-9 md:h-9"
                            }
                        }}
                    />
                </SignedIn>

                {/* History Button */}
                <button
                    onClick={() => router.push('/strikesense/history')}
                    className="flex items-center gap-1.5 md:gap-2 px-2.5 md:px-3 py-2 text-xs md:text-sm font-medium text-neutral-500 border border-neutral-200 rounded-full hover:bg-neutral-100 hover:text-black transition-all duration-300"
                >
                    <Clock className="w-3.5 h-3.5 md:w-4 md:h-4" />
                    <span className="hidden sm:inline">My Progress</span>
                </button>
            </div>
        </header>
    );
}
