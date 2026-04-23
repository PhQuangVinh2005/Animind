'use client';

import { ChatMessage, AnimePayload } from '@/types';
import MessageBubble from './MessageBubble';
import { useEffect, useRef, useState, useCallback } from 'react';
import { streamChat } from '@/lib/api';
import { loadMessages, saveMessages } from '@/lib/sessions';

interface ChatWindowProps {
  threadId: string;
  onFirstMessage: (text: string) => void;
}

// Simple ID generator (no dependency on uuid package)
function makeId() {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export default function ChatWindow({ threadId, onFirstMessage }: ChatWindowProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const isFirstMessage = useRef(true);

  // Restore messages when session changes (or on first mount)
  useEffect(() => {
    if (!threadId) return;
    const saved = loadMessages(threadId);
    setMessages(saved);
    // Only treat as first message if no history exists
    isFirstMessage.current = saved.length === 0;
    inputRef.current?.focus();
  }, [threadId]);

  // Auto-scroll on new content
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = useCallback(async () => {
    const text = input.trim();
    if (!text || isStreaming) return;

    setInput('');
    setIsStreaming(true);

    // Notify parent of first message for session title
    if (isFirstMessage.current) {
      onFirstMessage(text);
      isFirstMessage.current = false;
    }

    const userMsg: ChatMessage = { id: makeId(), role: 'user', content: text };
    const assistantId = makeId();
    const assistantMsg: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      streaming: true,
    };

    setMessages(prev => [...prev, userMsg, assistantMsg]);

    let pendingCards: AnimePayload[] = [];

    await streamChat(text, threadId, {
      onToken(token) {
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantId ? { ...m, content: m.content + token } : m,
          ),
        );
      },
      onCards(cards) {
        pendingCards = cards;
      },
      onDone() {
        setMessages(prev => {
          const updated = prev.map(m =>
            m.id === assistantId
              ? { ...m, streaming: false, cards: pendingCards.length ? pendingCards : undefined }
              : m,
          );
          saveMessages(threadId, updated);
          return updated;
        });
        setIsStreaming(false);
      },
      onError(msg) {
        setMessages(prev => {
          const updated = prev.map(m =>
            m.id === assistantId
              ? { ...m, content: `⚠️ ${msg}`, streaming: false, error: true }
              : m,
          );
          saveMessages(threadId, updated);
          return updated;
        });
        setIsStreaming(false);
      },
    });
  }, [input, isStreaming, threadId, onFirstMessage]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Auto-resize textarea
  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 scroll-smooth">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center select-none">
            <div className="text-5xl mb-4">🌸</div>
            <h2 className="text-xl font-semibold text-white drop-shadow mb-1">
              AniMind
            </h2>
            <p className="text-white/70 text-sm drop-shadow max-w-xs">
              Ask me anything about anime — recommendations, details, ratings, and more.
            </p>
            <div className="mt-6 flex flex-wrap gap-2 justify-center">
              {SUGGESTIONS.map(s => (
                <button
                  key={s}
                  onClick={() => setInput(s)}
                  className="text-xs bg-white/70 hover:bg-white/90 backdrop-blur-sm text-gray-700 rounded-full px-3 py-1.5 transition-all border border-white/50 shadow-sm"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map(msg => <MessageBubble key={msg.id} message={msg} />)
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div className="px-4 pb-4 pt-2">
        <div className="flex items-end gap-2 bg-white/80 backdrop-blur-md rounded-2xl border border-white/60 shadow-md px-4 py-2">
          <textarea
            ref={inputRef}
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder="Ask about anime..."
            rows={1}
            disabled={isStreaming}
            className="flex-1 resize-none bg-transparent text-gray-900 placeholder-gray-400 text-sm focus:outline-none disabled:opacity-50 leading-relaxed py-1 max-h-[120px]"
          />
          <button
            onClick={sendMessage}
            disabled={!input.trim() || isStreaming}
            className="flex-shrink-0 w-8 h-8 rounded-xl bg-sky-400 hover:bg-sky-500 disabled:bg-gray-200 disabled:cursor-not-allowed text-white flex items-center justify-center transition-all shadow-sm mb-0.5"
            aria-label="Send"
          >
            {isStreaming ? (
              <span className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              <svg viewBox="0 0 24 24" className="w-4 h-4 fill-current">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
              </svg>
            )}
          </button>
        </div>
        <p className="text-center text-[10px] text-white/40 mt-1.5">
          Press Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}

const SUGGESTIONS = [
  'Best action anime 2023?',
  'Tell me about Vinland Saga',
  'Anime with redemption theme',
  'Spy x Family details',
];
