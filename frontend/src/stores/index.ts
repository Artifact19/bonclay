import { create } from "zustand";
import type { Voice, Generation } from "@/types/api";
import type { Session, User } from "@supabase/supabase-js";

interface VoiceState {
  voices: Voice[];
  selectedVoiceId: string | null;
  setVoices: (v: Voice[]) => void;
  addVoice: (v: Voice) => void;
  removeVoice: (id: string) => void;
  selectVoice: (id: string | null) => void;
}

export const useVoiceStore = create<VoiceState>((set) => ({
  voices: [],
  selectedVoiceId: null,
  setVoices: (voices) =>
    set((s) => ({
      voices,
      selectedVoiceId: voices.some((v) => v.id === s.selectedVoiceId) ? s.selectedVoiceId : null,
    })),
  addVoice: (v) => set((s) => ({ voices: [v, ...s.voices.filter((x) => x.id !== v.id)] })),
  removeVoice: (id) =>
    set((s) => ({
      voices: s.voices.filter((v) => v.id !== id),
      selectedVoiceId: s.selectedVoiceId === id ? null : s.selectedVoiceId,
    })),
  selectVoice: (selectedVoiceId) => set({ selectedVoiceId }),
}));

export interface CurrentAudio {
  url: string;
  text?: string;
  voiceId?: string | null;
}

interface GenerationState {
  currentAudio: CurrentAudio | null;
  history: Generation[];
  isGenerating: boolean;
  progress: number;
  setCurrentAudio: (a: CurrentAudio | null) => void;
  setHistory: (h: Generation[]) => void;
  addToHistory: (g: Generation) => void;
  startGeneration: () => void;
  setProgress: (p: number) => void;
  finishGeneration: () => void;
  reset: () => void;
}

export const useGenerationStore = create<GenerationState>((set) => ({
  currentAudio: null,
  history: [],
  isGenerating: false,
  progress: 0,
  setCurrentAudio: (currentAudio) => set({ currentAudio }),
  setHistory: (history) => set({ history }),
  addToHistory: (g) => set((s) => ({ history: [g, ...s.history].slice(0, 100) })),
  startGeneration: () => set({ isGenerating: true, progress: 0 }),
  setProgress: (p) => set({ progress: Math.max(0, Math.min(100, p)) }),
  finishGeneration: () => set({ isGenerating: false, progress: 100 }),
  reset: () => set({ currentAudio: null, isGenerating: false, progress: 0 }),
}));

interface AuthState {
  user: User | null;
  session: Session | null;
  role: string;
  loading: boolean;
  setAuth: (session: Session | null) => void;
  setLoading: (l: boolean) => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  session: null,
  role: "user",
  loading: true,
  setAuth: (session) =>
    set({
      session,
      user: session?.user ?? null,
      role: (session?.user?.app_metadata?.role as string) || "user",
      loading: false,
    }),
  setLoading: (loading) => set({ loading }),
}));
