'use client';

import { AnimePayload, getTitle } from '@/types';
import Image from 'next/image';

interface AnimeCardProps {
  payload: AnimePayload;
}

function scoreColor(score?: number | null): string {
  if (!score) return 'bg-gray-100 text-gray-600';
  const s = score / 10; // raw score is out of 100
  if (s >= 8.0) return 'bg-green-100 text-green-700';
  if (s >= 6.0) return 'bg-yellow-100 text-yellow-700';
  return 'bg-red-100 text-red-700';
}

function formatDuration(mins?: number | null): string {
  if (!mins) return 'N/A';
  if (mins < 60) return `${mins}m`;
  return `${Math.floor(mins / 60)}h ${mins % 60}m`;
}

export default function AnimeCard({ payload }: AnimeCardProps) {
  const title = getTitle(payload);
  const scoreRaw = payload.score_display ?? (payload.score ? payload.score / 10 : null);
  const scoreLabel = scoreRaw ? scoreRaw.toFixed(1) : null;

  const season = payload.season_display
    ? `${payload.season_display} ${payload.year ?? ''}`
    : payload.year
    ? `${payload.year}`
    : null;

  const genreList = (payload.genres ?? []).slice(0, 4);
  const tagList = (payload.tags ?? []).slice(0, 5);
  const studios = (payload.studios ?? []).slice(0, 2).join(', ');

  return (
    <div className="flex gap-3 bg-white/80 backdrop-blur-sm rounded-xl border border-white/60 shadow-sm p-3 mt-2 hover:bg-white/90 transition-colors">
      {/* Cover image */}
      <div className="flex-shrink-0 w-[72px] h-[100px] rounded-lg overflow-hidden bg-gray-100 relative">
        {payload.cover_image ? (
          <Image
            src={payload.cover_image}
            alt={title}
            fill
            className="object-cover"
            sizes="72px"
            unoptimized
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-gray-300 text-xs text-center p-1">
            No image
          </div>
        )}
      </div>

      {/* Info */}
      <div className="flex-1 min-w-0 flex flex-col gap-1">
        {/* Title + score */}
        <div className="flex items-start justify-between gap-2">
          <a
            href={payload.site_url ?? '#'}
            target="_blank"
            rel="noopener noreferrer"
            className="font-semibold text-sm text-gray-900 hover:text-sky-600 transition-colors leading-tight line-clamp-2"
          >
            {title}
          </a>
          {scoreLabel && (
            <span
              className={`flex-shrink-0 text-xs font-bold px-2 py-0.5 rounded-full ${scoreColor(payload.score)}`}
            >
              ★ {scoreLabel}
            </span>
          )}
        </div>

        {/* Meta row */}
        <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs text-gray-500">
          {payload.format && (
            <span className="font-medium text-gray-700 uppercase text-[10px] bg-gray-100 px-1.5 py-0.5 rounded">
              {payload.format}
            </span>
          )}
          {season && <span>{season}</span>}
          {payload.episodes != null && (
            <span>{payload.episodes} ep{payload.episodes !== 1 ? 's' : ''}</span>
          )}
          {payload.duration != null && <span>{formatDuration(payload.duration)}</span>}
          {payload.popularity != null && (
            <span>👥 {payload.popularity.toLocaleString()}</span>
          )}
          {studios && <span>🎬 {studios}</span>}
        </div>

        {/* Genres */}
        {genreList.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {genreList.map(g => (
              <span
                key={g}
                className="text-[10px] px-1.5 py-0.5 rounded-full bg-sky-100 text-sky-700 font-medium"
              >
                {g}
              </span>
            ))}
          </div>
        )}

        {/* Tags */}
        {tagList.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {tagList.map(t => (
              <span
                key={t}
                className="text-[10px] px-1.5 py-0.5 rounded-full bg-gray-100 text-gray-500"
              >
                {t}
              </span>
            ))}
          </div>
        )}

        {/* AniList link */}
        {payload.site_url && (
          <a
            href={payload.site_url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-[10px] text-sky-500 hover:text-sky-700 transition-colors mt-auto pt-0.5 flex items-center gap-0.5"
          >
            View on AniList ↗
          </a>
        )}
      </div>
    </div>
  );
}
