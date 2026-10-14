import React, { useState } from 'react';
import {
  FileText,
  Trash2,
  CheckSquare,
  Square,
  ChevronDown,
  ChevronRight,
  Upload,
} from 'lucide-react';
import { DocumentItem } from '../../types';

export interface DocumentListProps {
  documents: DocumentItem[];
  selectedDocIds: string[];
  isLoading?: boolean;
  onToggleSelect: (id: string) => void;
  onSelectAll: () => void;
  onClearSelection: () => void;
  onDeleteDocument: (id: string) => void;
  onOpenUpload: () => void;
  isCollapsed?: boolean;
  onToggleCollapse?: () => void;
  className?: string;
}

export const DocumentList: React.FC<DocumentListProps> = ({
  documents,
  selectedDocIds,
  isLoading = false,
  onToggleSelect,
  onSelectAll,
  onClearSelection,
  onDeleteDocument,
  onOpenUpload,
  isCollapsed,
  onToggleCollapse,
  className = '',
}) => {
  const [internalExpanded, setInternalExpanded] = useState<boolean>(true);
  const isExpanded = isCollapsed !== undefined ? !isCollapsed : internalExpanded;

  const toggleExpanded = (): void => {
    if (onToggleCollapse) {
      onToggleCollapse();
    } else {
      setInternalExpanded((prev) => !prev);
    }
  };

  const allSelected = documents.length > 0 && selectedDocIds.length === documents.length;

  return (
    <div
      className={`flex flex-col bg-[#141417] select-none text-zinc-200 border-b border-zinc-800/80 ${
        isExpanded ? 'flex-1 min-h-0' : 'shrink-0'
      } ${className}`}
    >
      {/* Sleek Integrated Section Header (32px) */}
      <div className="h-8 px-2.5 border-b border-zinc-800/60 flex items-center justify-between bg-[#121215] shrink-0">
        <button
          type="button"
          onClick={toggleExpanded}
          aria-label="Toggle Corpus Explorer"
          aria-expanded={isExpanded}
          className="flex items-center space-x-1.5 text-left focus:outline-hidden cursor-pointer group min-w-0"
        >
          {isExpanded ? (
            <ChevronDown className="w-3.5 h-3.5 text-zinc-400 group-hover:text-zinc-200 transition-colors shrink-0" />
          ) : (
            <ChevronRight className="w-3.5 h-3.5 text-zinc-400 group-hover:text-zinc-200 transition-colors shrink-0" />
          )}
          <span className="text-[10px] font-mono font-semibold uppercase tracking-wider text-zinc-400 group-hover:text-zinc-200 transition-colors shrink-0">
            Corpus Explorer
          </span>
          {documents.length > 0 && (
            <span className="text-[10px] font-mono text-zinc-400 font-normal truncate">
              ({documents.length}) {selectedDocIds.length === 0
                ? 'Scope: All Documents'
                : `Scoped: ${selectedDocIds.length} of ${documents.length} doc(s)`}
            </span>
          )}
        </button>

        {/* Master Scoping Actions in Header */}
        {documents.length > 0 && isExpanded && (
          <div className="flex items-center space-x-1 text-[10px] font-mono shrink-0">
            <button
              type="button"
              onClick={onSelectAll}
              disabled={allSelected}
              className="text-zinc-400 hover:text-indigo-400 disabled:opacity-30 transition cursor-pointer"
            >
              Select All
            </button>
            <span className="text-zinc-600">·</span>
            <button
              type="button"
              onClick={onClearSelection}
              disabled={selectedDocIds.length === 0}
              className="text-zinc-400 hover:text-indigo-400 disabled:opacity-30 transition cursor-pointer"
            >
              Clear
            </button>
          </div>
        )}
      </div>

      {/* Collapsible Body */}
      {isExpanded && (
        <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
          {/* Main List Area: Flat Tree (Cursor/Zed style) */}
          <div className="flex-1 overflow-y-auto px-1.5 py-1 space-y-0.5">
            {isLoading ? (
              <div data-testid="document-list-skeleton" className="space-y-1.5 p-1.5">
                {[1, 2, 3].map((n) => (
                  <div
                    key={n}
                    className="h-7 rounded-md bg-zinc-800/40 animate-pulse flex items-center px-2 space-x-2"
                  >
                    <div className="w-3 h-3 bg-zinc-700/60 rounded-xs" />
                    <div className="h-2.5 bg-zinc-700/60 rounded-xs flex-1" />
                    <div className="w-12 h-2 bg-zinc-700/40 rounded-xs" />
                  </div>
                ))}
              </div>
            ) : documents.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center p-4 text-center text-zinc-400">
                <FileText className="w-6 h-6 text-zinc-500 mb-2 stroke-[1.5]" />
                <p className="text-xs font-medium text-zinc-300">No documents indexed</p>
                <p className="text-[11px] text-zinc-500 mt-0.5 max-w-[200px] leading-relaxed">
                  Upload technical PDFs to extract structure and enable conversational RAG.
                </p>
                <button
                  type="button"
                  onClick={onOpenUpload}
                  className="mt-3 inline-flex items-center px-2.5 py-1 text-xs font-medium rounded-md bg-indigo-600 hover:bg-indigo-500 text-white transition cursor-pointer"
                  aria-label="Upload PDF"
                >
                  <Upload className="w-3 h-3 mr-1" />
                  <span>Upload PDF</span>
                </button>
              </div>
            ) : (
              documents.map((doc) => {
                const isSelected = selectedDocIds.includes(doc.id);
                const pageCount = doc.pageCount ?? doc.page_count ?? 0;
                const chunkCount = doc.chunkCount ?? doc.chunk_count ?? 0;

                return (
                  <div
                    key={doc.id}
                    onClick={() => onToggleSelect(doc.id)}
                    className={`group relative px-2 py-1.5 rounded-md text-xs transition-colors duration-100 cursor-pointer flex items-center justify-between gap-1.5 ${
                      isSelected
                        ? 'bg-indigo-950/25 text-zinc-100'
                        : 'text-zinc-300 hover:bg-zinc-800/40 hover:text-zinc-100'
                    }`}
                  >
                    {/* Left: Checkbox + Status Dot + Filename */}
                    <div className="flex items-center space-x-1.5 min-w-0 flex-1">
                      <button
                        type="button"
                        aria-label="Toggle selection"
                        onClick={(e) => {
                          e.stopPropagation();
                          onToggleSelect(doc.id);
                        }}
                        className="text-zinc-500 hover:text-indigo-400 shrink-0 cursor-pointer"
                      >
                        {isSelected ? (
                          <CheckSquare className="w-3.5 h-3.5 text-indigo-400" />
                        ) : (
                          <Square className="w-3.5 h-3.5 text-zinc-600 group-hover:text-zinc-400" />
                        )}
                      </button>

                      <span
                        className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                          doc.status === 'ready'
                            ? 'bg-emerald-400/80 shadow-[0_0_4px_rgba(52,211,153,0.3)]'
                            : doc.status === 'processing'
                            ? 'bg-amber-400'
                            : 'bg-rose-400'
                        }`}
                        title={doc.status}
                      />

                      <span
                        className="truncate text-xs font-normal text-zinc-300 group-hover:text-zinc-100"
                        title={doc.filename}
                      >
                        {doc.filename}
                      </span>
                    </div>

                    {/* Right: Unboxed Monospace Telemetry + Hover Delete */}
                    <div className="flex items-center space-x-1.5 shrink-0">
                      <span
                        className="text-[10px] font-mono text-zinc-400 transition-colors"
                        title={`${pageCount} pages, ${chunkCount} chunks`}
                      >
                        p. {pageCount} · {chunkCount} chk
                      </span>

                      <button
                        type="button"
                        aria-label="Delete document"
                        title="Delete document"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteDocument(doc.id);
                        }}
                        className="p-0.5 rounded text-zinc-500 hover:text-rose-400 opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
      )}
    </div>
  );
};
