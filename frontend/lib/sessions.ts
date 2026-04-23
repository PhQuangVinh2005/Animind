import { StoredSession, ChatMessage } from '@/types';

const SESSIONS_KEY = 'animind_sessions';
const MSG_PREFIX   = 'animind_msgs_';   // per-thread message store

export function getSessions(): StoredSession[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = localStorage.getItem(SESSIONS_KEY);
    return raw ? (JSON.parse(raw) as StoredSession[]) : [];
  } catch {
    return [];
  }
}

export function addSession(session: StoredSession): void {
  if (typeof window === 'undefined') return;
  const sessions = getSessions().filter(s => s.thread_id !== session.thread_id);
  sessions.unshift(session); // newest first
  localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions.slice(0, 50)));
}

export function removeSession(thread_id: string): void {
  if (typeof window === 'undefined') return;
  const sessions = getSessions().filter(s => s.thread_id !== thread_id);
  localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
  localStorage.removeItem(MSG_PREFIX + thread_id);
}

export function generateThreadId(): string {
  // UUID v4
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

// ── Per-thread message persistence ───────────────────────────────────────────

/** Save all completed messages for a thread (strips streaming flag). */
export function saveMessages(thread_id: string, messages: ChatMessage[]): void {
  if (typeof window === 'undefined' || !thread_id) return;
  // Only persist messages that are no longer streaming
  const completed = messages
    .filter(m => !m.streaming)
    .map(m => ({ ...m, streaming: false }));
  try {
    localStorage.setItem(MSG_PREFIX + thread_id, JSON.stringify(completed));
  } catch {
    // localStorage quota exceeded — silently ignore
  }
}

/** Load persisted messages for a thread, or [] if none. */
export function loadMessages(thread_id: string): ChatMessage[] {
  if (typeof window === 'undefined' || !thread_id) return [];
  try {
    const raw = localStorage.getItem(MSG_PREFIX + thread_id);
    return raw ? (JSON.parse(raw) as ChatMessage[]) : [];
  } catch {
    return [];
  }
}
