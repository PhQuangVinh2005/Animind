'use client';

import { useCallback, useEffect, useState } from 'react';
import Sidebar from '@/components/Sidebar';
import ChatWindow from '@/components/ChatWindow';
import { StoredSession } from '@/types';
import { addSession, generateThreadId, getSessions } from '@/lib/sessions';

export default function Home() {
  const [sessions, setSessions] = useState<StoredSession[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string>('');

  // Init on mount
  useEffect(() => {
    const stored = getSessions();
    setSessions(stored);

    // Resume last session or create new
    if (stored.length > 0) {
      setActiveThreadId(stored[0].thread_id);
    } else {
      setActiveThreadId(generateThreadId());
    }
  }, []);

  const refreshSessions = useCallback(() => {
    setSessions(getSessions());
  }, []);

  const handleNewChat = useCallback(() => {
    const id = generateThreadId();
    setActiveThreadId(id);
    refreshSessions();
  }, [refreshSessions]);

  const handleSelectSession = useCallback((threadId: string) => {
    setActiveThreadId(threadId);
  }, []);

  // Called by ChatWindow with the first message text → create session title
  const handleFirstMessage = useCallback(
    (text: string) => {
      const title = text.length > 45 ? text.slice(0, 42) + '…' : text;
      const newSession: StoredSession = {
        thread_id: activeThreadId,
        title,
        created_at: Date.now(),
      };
      addSession(newSession);
      refreshSessions();
    },
    [activeThreadId, refreshSessions],
  );

  if (!activeThreadId) return null; // hydration guard

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        sessions={sessions}
        activeThreadId={activeThreadId}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
        onSessionsChange={refreshSessions}
      />
      <main className="flex-1 flex flex-col overflow-hidden">
        <ChatWindow
          key={activeThreadId}
          threadId={activeThreadId}
          onFirstMessage={handleFirstMessage}
        />
      </main>
    </div>
  );
}
