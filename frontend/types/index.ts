// ─── Anime payload from backend ───────────────────────────────────────────────

export interface AnimeTitle {
  preferred: string;
  english?: string | null;
  native?: string | null;
}

export interface AnimePayload {
  anilist_id: number;
  mal_id?: number;
  title: AnimeTitle | string;
  cover_image?: string | null;
  banner_image?: string | null;
  site_url?: string | null;
  year?: number | null;
  season?: string | null;
  season_display?: string | null;
  genres?: string[];
  tags?: string[];
  studios?: string[];
  format?: string | null;
  status?: string | null;
  score?: number | null;
  score_display?: number | null;
  episodes?: number | null;
  duration?: number | null;
  popularity?: number | null;
  // tool_output "detail" shape
  found?: boolean;
  anime?: AnimePayload;
  // top-level title might also be a string
  preferred_title?: string;
}

export function getTitle(payload: AnimePayload): string {
  if (typeof payload.title === 'object' && payload.title !== null) {
    return payload.title.preferred || payload.title.english || 'Unknown';
  }
  if (typeof payload.title === 'string') return payload.title;
  if (payload.preferred_title) return payload.preferred_title;
  return 'Unknown';
}

// ─── Chat / Session ────────────────────────────────────────────────────────────

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  cards?: AnimePayload[];
  streaming?: boolean;
  error?: boolean;
}

export interface StoredSession {
  thread_id: string;
  title: string;
  created_at: number;
}
