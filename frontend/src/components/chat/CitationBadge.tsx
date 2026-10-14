import React from 'react';
import { FileText } from 'lucide-react';
import { CitationDetail } from '../../types';

export interface CitationBadgeProps {
  citation: CitationDetail;
  onSelect?: (citation: CitationDetail) => void;
  className?: string;
}

export function CitationBadge({
  citation,
  onSelect,
  className = '',
}: CitationBadgeProps): React.JSX.Element {
  const { citationIndex, filename, pageNumber, sectionTitle, sourceText } = citation;

  const truncateFilename = (name: string, maxLen = 18): string => {
    if (!name) return '';
    if (name.length <= maxLen) return name;
    const parts = name.split('.');
    const ext = parts.length > 1 ? `.${parts.pop()}` : '';
    const base = parts.join('.');
    return `${base.slice(0, maxLen - ext.length - 3)}...${ext}`;
  };

  const displayName = filename ? truncateFilename(filename) : '';
  const tooltipText = filename
    ? `${filename} • Page ${pageNumber}${sectionTitle ? ` • ${sectionTitle}` : ''}\n"${sourceText.slice(0, 120)}..."`
    : `Citation [${citationIndex}] • Page ${pageNumber}`;

  return (
    <button
      type="button"
      onClick={() => onSelect?.(citation)}
      title={tooltipText}
      aria-label={`Citation ${citationIndex}: ${filename || 'document'}, page ${pageNumber}`}
      className={`group inline-flex items-center space-x-1.5 px-2 py-0.5 mx-0.5 rounded-md text-[11px] font-mono font-medium bg-zinc-800/90 text-zinc-300 border border-zinc-700/80 hover:bg-zinc-700/90 hover:text-white hover:border-indigo-500/50 focus:outline-none focus:ring-1 focus:ring-indigo-500/50 transition-all select-none align-baseline shadow-2xs active:scale-95 cursor-pointer ${className}`}
    >
      <FileText className="w-3 h-3 text-indigo-400 group-hover:text-indigo-300 transition-colors shrink-0" />
      <span className="tracking-tight">
        {displayName ? `[${citationIndex}: ${displayName}, p.${pageNumber}]` : `[${citationIndex}]`}
      </span>
    </button>
  );
}
