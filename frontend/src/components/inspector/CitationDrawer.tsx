import React, { useState, useEffect } from 'react';
import {
  X,
  FileText,
  ChevronLeft,
  ChevronRight,
  Copy,
  Check,
  Bookmark,
  Layers,
  Hash,
} from 'lucide-react';
import { CitationDetail } from '../../types';

export interface CitationDrawerProps {
  isOpen: boolean;
  citation: CitationDetail | null;
  citations?: CitationDetail[];
  onClose: () => void;
  onSelectCitation?: (citation: CitationDetail) => void;
  className?: string;
}

export function CitationDrawer({
  isOpen,
  citation,
  citations = [],
  onClose,
  onSelectCitation,
  className = '',
}: CitationDrawerProps): React.JSX.Element | null {
  const [copied, setCopied] = useState<boolean>(false);

  // Close on Escape key press
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, onClose]);

  if (!isOpen || !citation) {
    return null;
  }

  // Derive citation list and current index
  const citationList = citations.length > 0 ? citations : [citation];
  const currentIndex = citationList.findIndex(
    (c) =>
      c.citationIndex === citation.citationIndex ||
      (c.chunkId && c.chunkId === citation.chunkId)
  );
  const activeIndex = currentIndex >= 0 ? currentIndex : 0;
  const totalCitations = citationList.length;

  const handlePrev = (): void => {
    if (totalCitations <= 1) return;
    const nextIdx = (activeIndex - 1 + totalCitations) % totalCitations;
    onSelectCitation?.(citationList[nextIdx]);
  };

  const handleNext = (): void => {
    if (totalCitations <= 1) return;
    const nextIdx = (activeIndex + 1) % totalCitations;
    onSelectCitation?.(citationList[nextIdx]);
  };

  const handleCopyExcerpt = async (): Promise<void> => {
    const textToCopy = citation.sourceText || citation.quote || '';
    if (!textToCopy) return;

    try {
      await navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Ignore clipboard write failures in test/restricted environments
    }
  };

  const truncateName = (name: string, maxLen = 22): string => {
    if (!name) return 'document.pdf';
    if (name.length <= maxLen) return name;
    const parts = name.split('.');
    const ext = parts.length > 1 ? `.${parts.pop()}` : '';
    const base = parts.join('.');
    return `${base.slice(0, maxLen - ext.length - 3)}...${ext}`;
  };

  const excerptText = citation.sourceText || citation.quote || 'No excerpt text provided.';

  return (
    <>
      {/* Backdrop for mobile/tablet viewports only (hidden on desktop to allow side-by-side reading) */}
      <div
        data-testid="citation-drawer-backdrop"
        onClick={onClose}
        className="fixed inset-0 bg-black/70 backdrop-blur-xs z-40 xl:hidden transition-opacity animate-in fade-in duration-200"
      />

      {/* Pane 3: Persistent Docked Workbench Panel on desktop (xl), Slide-over Drawer on mobile */}
      <aside
        role="dialog"
        aria-modal="true"
        aria-label="Citation Inspector"
        data-testid="evidence-grounding-pane"
        className={`fixed right-0 top-0 bottom-0 h-full w-full sm:max-w-lg md:max-w-xl z-50 shadow-2xl xl:static xl:z-20 xl:w-[420px] xl:min-w-[400px] xl:max-w-[460px] xl:shrink-0 xl:shadow-none bg-[#09090b] text-zinc-100 border-l border-zinc-800 flex flex-col justify-between overflow-hidden animate-in slide-in-from-right duration-200 ${className}`}
      >
        {/* Top Header */}
        <div className="border-b border-zinc-800 bg-[#141417] px-4 pt-3.5 pb-3 shrink-0">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono font-semibold tracking-wider uppercase bg-indigo-500/10 text-indigo-400 border border-indigo-500/30">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse mr-1.5" />
                Citation & Attribution Grounding
              </span>
              {totalCitations > 1 && (
                <span className="text-[11px] font-mono text-zinc-400 bg-zinc-800/80 px-1.5 py-0.5 rounded border border-zinc-700">
                  {activeIndex + 1} of {totalCitations}
                </span>
              )}
            </div>

            <div className="flex items-center space-x-2">
              <span className="hidden sm:inline-block font-mono text-[10px] text-zinc-400 border border-zinc-700/60 px-1.5 py-0.5 rounded bg-zinc-900">
                ESC
              </span>
              <button
                type="button"
                onClick={onClose}
                aria-label="Close Inspector"
                className="p-1 rounded text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800 transition-colors cursor-pointer"
                title="Close drawer (Esc)"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="p-1.5 rounded-md bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
                <FileText className="w-4 h-4" />
              </div>
              <h2 className="font-semibold text-sm text-zinc-100 tracking-tight">
                Citation Inspector
              </h2>
            </div>

            {/* Previous / Next Citation Switcher Controls */}
            {totalCitations > 1 && (
              <div className="flex items-center space-x-1 bg-zinc-900/90 p-0.5 rounded-md border border-zinc-800">
                <button
                  type="button"
                  onClick={handlePrev}
                  aria-label="Previous citation"
                  className="px-2 py-1 rounded text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800 transition-colors flex items-center space-x-1 font-mono text-xs cursor-pointer"
                  title="Previous citation"
                >
                  <ChevronLeft className="w-3.5 h-3.5" />
                  <span className="hidden sm:inline">Prev</span>
                </button>
                <div className="w-px h-3.5 bg-zinc-800" />
                <button
                  type="button"
                  onClick={handleNext}
                  aria-label="Next citation"
                  className="px-2 py-1 rounded text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800 transition-colors flex items-center space-x-1 font-mono text-xs cursor-pointer"
                  title="Next citation"
                >
                  <span className="hidden sm:inline">Next</span>
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
            )}
          </div>

          {/* Citation Carousel Tabs */}
          {totalCitations > 1 && (
            <div className="flex items-center space-x-2 mt-2.5 overflow-x-auto pb-1 text-xs font-mono scrollbar-thin">
              {citationList.map((c, i) => {
                const isActive = i === activeIndex;
                const citNum = c.citationIndex || i + 1;
                const docName = truncateName(c.filename, 16);
                return (
                  <button
                    key={`cit_tab_${c.chunkId || i}_${citNum}`}
                    type="button"
                    onClick={() => onSelectCitation?.(c)}
                    aria-label={`Citation ${citNum}: ${c.filename || 'document'}`}
                    className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-md transition-all shrink-0 cursor-pointer ${
                      isActive
                        ? 'bg-zinc-800 border border-indigo-500/50 text-indigo-300 shadow-xs'
                        : 'bg-zinc-900/70 hover:bg-zinc-800/60 border border-zinc-800 text-zinc-400'
                    }`}
                  >
                    <span
                      className={`text-[11px] font-bold px-1 rounded ${
                        isActive
                          ? 'bg-indigo-600 text-white'
                          : 'bg-zinc-800 text-zinc-300'
                      }`}
                    >
                      {citNum}
                    </span>
                    <span className="truncate max-w-[110px] text-[11px]">{docName}</span>
                    <span className="text-[10px] text-zinc-500">p.{c.pageNumber}</span>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Scrollable Inspector Body */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3.5">
          {/* Active Attribution Header Card */}
          <div className="rounded-lg bg-[#141417] border border-zinc-800 p-3.5 space-y-2.5">
            <div className="flex items-start justify-between">
              <div className="flex items-start space-x-2.5 min-w-0">
                <div className="p-1.5 rounded-md bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 shrink-0">
                  <FileText className="w-4 h-4" />
                </div>
                <div className="min-w-0">
                  <h3 className="font-medium text-xs text-zinc-100 truncate">
                    {citation.filename || 'Untitled Document'}
                  </h3>
                  <p className="text-[11px] text-zinc-400 font-mono mt-0.5">
                    Citation [{citation.citationIndex}] • Page {citation.pageNumber}
                  </p>
                </div>
              </div>

              <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-emerald-950/60 text-emerald-400 border border-emerald-800/60 shrink-0">
                Grounded
              </span>
            </div>

            {/* Attribution Tags */}
            <div className="flex flex-wrap items-center gap-1.5 pt-2 border-t border-zinc-800/80 text-xs">
              <div className="inline-flex items-center space-x-1 px-2 py-0.5 rounded bg-[#18181c] text-zinc-300 border border-zinc-800 font-mono text-[11px]">
                <Bookmark className="w-3 h-3 text-indigo-400" />
                <span>Page {citation.pageNumber}</span>
              </div>

              {citation.sectionTitle && (
                <div className="inline-flex items-center space-x-1 px-2 py-0.5 rounded bg-[#18181c] text-zinc-300 border border-zinc-800 text-[11px]">
                  <span className="text-zinc-500 text-xs font-semibold">§</span>
                  <span className="truncate max-w-[220px]">{citation.sectionTitle}</span>
                </div>
              )}

              {citation.chunkId && (
                <div className="inline-flex items-center space-x-1 px-2 py-0.5 rounded bg-[#18181c] text-zinc-400 border border-zinc-800 font-mono text-[10px]">
                  <Hash className="w-3 h-3 text-zinc-500" />
                  <span className="truncate max-w-[120px]">{citation.chunkId}</span>
                </div>
              )}
            </div>
          </div>

          {/* Source Excerpt Card */}
          <div className="rounded-lg bg-[#0e0e11] border border-zinc-800 overflow-hidden shadow-xs">
            {/* Excerpt Header */}
            <div className="bg-[#141417] px-3.5 py-2 border-b border-zinc-800 flex items-center justify-between">
              <span className="text-[10px] font-mono font-semibold text-zinc-300 uppercase tracking-wider">
                Original Source Excerpt
              </span>

              <button
                type="button"
                onClick={() => void handleCopyExcerpt()}
                aria-label="Copy Excerpt"
                className="flex items-center space-x-1 px-2 py-0.5 rounded bg-zinc-800 hover:bg-zinc-700 text-xs font-mono text-zinc-200 border border-zinc-700/60 transition-colors cursor-pointer"
                title="Copy source passage to clipboard"
              >
                {copied ? (
                  <>
                    <Check className="w-3 h-3 text-emerald-400" />
                    <span className="text-emerald-400 font-medium">Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3 h-3 text-zinc-400" />
                    <span>Copy Excerpt</span>
                  </>
                )}
              </button>
            </div>

            {/* Excerpt Text Passage */}
            <div className="p-3.5 border-l-2 border-l-indigo-500/70">
              <pre className="font-mono text-xs text-zinc-200 leading-relaxed whitespace-pre-wrap select-text break-words">
                {excerptText}
              </pre>
            </div>
          </div>

          {/* Telemetry Details */}
          <div className="rounded-lg bg-[#141417] border border-zinc-800 p-3 space-y-2">
            <h4 className="text-[10px] font-mono font-semibold uppercase tracking-wider text-zinc-400 flex items-center space-x-1.5">
              <Layers className="w-3.5 h-3.5 text-indigo-400" />
              <span>Attribution Provenance</span>
            </h4>
            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              <div className="bg-[#09090b] p-2 rounded border border-zinc-800">
                <span className="text-zinc-500 text-[10px] block font-semibold">DOCUMENT ID</span>
                <span className="text-zinc-300 truncate block mt-0.5 text-[11px]">
                  {citation.documentId || 'doc_unknown'}
                </span>
              </div>
              <div className="bg-[#09090b] p-2 rounded border border-zinc-800">
                <span className="text-zinc-500 text-[10px] block font-semibold">CHUNK ID</span>
                <span className="text-zinc-300 truncate block mt-0.5 text-[11px]">
                  {citation.chunkId || `chunk_${citation.citationIndex}`}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Drawer Footer */}
        <div className="border-t border-zinc-800 bg-[#141417] px-4 py-2.5 shrink-0 flex items-center justify-between text-xs text-zinc-400">
          <span className="font-mono text-[11px] text-zinc-500 truncate max-w-[260px]">
            Page {citation.pageNumber} in {citation.filename || 'source document'}
          </span>
          <button
            type="button"
            onClick={onClose}
            className="px-2.5 py-1 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-xs font-medium transition cursor-pointer"
          >
            Close
          </button>
        </div>
      </aside>
    </>
  );
}
