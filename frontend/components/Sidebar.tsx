'use client';

import { StoredSession } from '@/types';
import { removeSession } from '@/lib/sessions';

interface SidebarProps {
  sessions: StoredSession[];
  activeThreadId: string;
  onSelectSession: (threadId: string) => void;
  onNewChat: () => void;
  onSessionsChange: () => void;
}

export default function Sidebar({
  sessions,
  activeThreadId,
  onSelectSession,
  onNewChat,
  onSessionsChange,
}: SidebarProps) {
  const handleDelete = (e: React.MouseEvent, threadId: string) => {
    e.stopPropagation();
    removeSession(threadId);
    onSessionsChange();
    if (activeThreadId === threadId) {
      onNewChat();
    }
  };

  return (
    <aside className="w-64 flex-shrink-0 flex flex-col bg-white/75 backdrop-blur-md border-r border-white/50 shadow-lg">
      {/* Logo */}
      <div className="px-4 pt-5 pb-3 border-b border-white/40">
        <div className="flex items-center gap-2 mb-3">
          <span className="text-2xl">🌸</span>
          <div>
            <h1 className="text-base font-bold text-gray-900 leading-tight">AniMind</h1>
            <p className="text-[10px] text-gray-400 leading-tight">Anime RAG Chatbot</p>
          </div>
        </div>

        {/* New Chat */}
        <button
          onClick={onNewChat}
          className="w-full flex items-center gap-2 px-3 py-2 rounded-xl bg-sky-400 hover:bg-sky-500 text-white text-sm font-medium transition-all shadow-sm"
        >
          <svg viewBox="0 0 24 24" className="w-4 h-4 fill-current flex-shrink-0">
            <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z" />
          </svg>
          New Chat
        </button>
      </div>

      {/* Session list */}
      <nav className="flex-1 overflow-y-auto py-2 px-2">
        {sessions.length === 0 ? (
          <p className="text-[11px] text-gray-400 text-center mt-4 px-2">
            No sessions yet.{' '}
            <br />
            Start a conversation!
          </p>
        ) : (
          sessions.map(session => {
            const isActive = session.thread_id === activeThreadId;
            return (
              <button
                key={session.thread_id}
                onClick={() => onSelectSession(session.thread_id)}
                className={`group w-full text-left px-3 py-2 rounded-lg mb-0.5 flex items-center gap-2 text-sm transition-all ${
                  isActive
                    ? 'bg-sky-100/80 text-sky-800 font-medium'
                    : 'text-gray-700 hover:bg-white/60'
                }`}
              >
                <span className="text-base flex-shrink-0">💬</span>
                <span className="flex-1 truncate text-xs leading-snug">
                  {session.title}
                </span>
                {/* Delete button */}
                <span
                  role="button"
                  tabIndex={0}
                  onClick={e => handleDelete(e, session.thread_id)}
                  onKeyDown={e => e.key === 'Enter' && handleDelete(e as unknown as React.MouseEvent, session.thread_id)}
                  className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-500 flex-shrink-0 transition-all p-0.5 rounded"
                  aria-label="Delete session"
                >
                  <svg viewBox="0 0 24 24" className="w-3 h-3 fill-current">
                    <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z" />
                  </svg>
                </span>
              </button>
            );
          })
        )}
      </nav>

      {/* Footer — GitHub */}
      <div className="px-4 py-3 border-t border-white/40">
        <a
          href="https://github.com/PhQuangVinh2005/Animind"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 px-3 py-2 rounded-xl hover:bg-white/60 text-gray-700 hover:text-gray-900 transition-all text-sm"
        >
          {/* GitHub SVG */}
          <svg viewBox="0 0 24 24" className="w-4 h-4 flex-shrink-0 fill-current">
            <path d="M12 0C5.374 0 0 5.373 0 12c0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0 1 12 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z" />
          </svg>
          <span className="text-xs">GitHub</span>
        </a>
      </div>
    </aside>
  );
}
