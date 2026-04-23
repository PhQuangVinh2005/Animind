'use client';

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ChatMessage } from '@/types';
import AnimeCard from './AnimeCard';

interface MessageBubbleProps {
  message: ChatMessage;
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-3`}>
      {/* Assistant avatar */}
      {!isUser && (
        <div className="flex-shrink-0 w-7 h-7 rounded-full bg-sky-400/80 flex items-center justify-center text-white text-xs font-bold mr-2 mt-1 shadow">
          A
        </div>
      )}

      <div className={`max-w-[82%] ${isUser ? 'items-end' : 'items-start'} flex flex-col`}>
        {/* Bubble */}
        <div
          className={
            isUser
              ? 'bg-sky-400/90 text-white rounded-2xl rounded-tr-sm px-4 py-3 shadow-sm'
              : message.error
              ? 'bg-red-50/90 text-red-700 border border-red-200 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm'
              : 'bg-white/85 backdrop-blur-sm text-gray-900 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm border border-white/60'
          }
        >
          {isUser ? (
            // User messages: plain text
            <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
          ) : (
            // Assistant messages: full markdown rendering
            <div className="prose prose-sm max-w-none prose-gray
              prose-p:my-1 prose-p:leading-relaxed
              prose-headings:font-semibold prose-headings:text-gray-900
              prose-h1:text-base prose-h2:text-sm prose-h3:text-sm
              prose-strong:font-semibold prose-strong:text-gray-900
              prose-em:text-gray-700
              prose-ul:my-1 prose-ul:pl-4 prose-ul:space-y-0.5
              prose-ol:my-1 prose-ol:pl-4 prose-ol:space-y-0.5
              prose-li:my-0 prose-li:leading-relaxed
              prose-code:bg-sky-50 prose-code:text-sky-700 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:text-xs prose-code:font-mono
              prose-pre:bg-gray-900 prose-pre:text-gray-100 prose-pre:rounded-lg prose-pre:text-xs
              prose-blockquote:border-sky-300 prose-blockquote:text-gray-600 prose-blockquote:not-italic
              prose-hr:border-gray-200
              prose-a:text-sky-600 prose-a:no-underline hover:prose-a:underline
            ">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
              {message.streaming && (
                <span className="inline-block w-0.5 h-4 bg-gray-600 ml-0.5 align-middle animate-pulse" />
              )}
            </div>
          )}
        </div>

        {/* Anime Cards — shown after streaming completes */}
        {!message.streaming && message.cards && message.cards.length > 0 && (
          <div className="w-full mt-1 space-y-1">
            {message.cards.map((card, i) => (
              <AnimeCard key={card.anilist_id ?? i} payload={card} />
            ))}
          </div>
        )}
      </div>

      {/* User avatar */}
      {isUser && (
        <div className="flex-shrink-0 w-7 h-7 rounded-full bg-sky-600/80 flex items-center justify-center text-white text-xs font-bold ml-2 mt-1 shadow">
          U
        </div>
      )}
    </div>
  );
}
