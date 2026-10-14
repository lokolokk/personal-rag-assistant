import React, { useEffect } from 'react';
import { X, Cpu, Terminal, ShieldCheck, Layers, Keyboard } from 'lucide-react';

export interface SystemSpecsModalProps {
  isOpen: boolean;
  onClose: () => void;
  selectedDocCount: number;
  totalDocCount?: number;
}

export function SystemSpecsModal({
  isOpen,
  onClose,
  selectedDocCount,
}: SystemSpecsModalProps): React.JSX.Element | null {
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

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        data-testid="system-specs-backdrop"
        onClick={onClose}
        className="fixed inset-0 bg-black/70 backdrop-blur-xs transition-opacity animate-in fade-in"
      />

      {/* Modal Dialog Card */}
      <div
        role="dialog"
        aria-modal="true"
        aria-label="System Architecture & Telemetry"
        className="relative w-full max-w-2xl bg-[#141417] border border-zinc-800 rounded-xl shadow-2xl z-10 overflow-hidden flex flex-col animate-in zoom-in-95 duration-150"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-800 bg-[#16161a]">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              <Cpu className="w-4 h-4" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h3 className="text-sm font-semibold text-zinc-100 tracking-tight">
                  System Architecture &amp; Telemetry
                </h3>
                <span className="px-1.5 py-0.5 rounded text-[10px] font-mono uppercase bg-indigo-500/10 text-indigo-400 border border-indigo-500/30">
                  Dual-Layer RAG
                </span>
              </div>
              <p className="text-xs text-zinc-400 font-sans mt-0.5">
                Engine specifications, retrieval pipeline parameters, and grounding protocol
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800/80 transition cursor-pointer"
            aria-label="Close system specs"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content Body */}
        <div className="p-5 space-y-4 max-h-[75vh] overflow-y-auto font-sans text-xs">
          {/* Subsystem 1: Hybrid Retrieval */}
          <div className="p-3.5 rounded-lg bg-[#111114] border border-zinc-800/90 space-y-2">
            <div className="flex items-center space-x-2 text-zinc-200 font-medium">
              <Terminal className="w-3.5 h-3.5 text-indigo-400" />
              <span>Hybrid Dense + Sparse RRF</span>
            </div>
            <p className="text-zinc-400 leading-relaxed">
              Combines dense semantic vector search via <span className="font-mono text-zinc-200">ChromaDB</span> (<span className="font-mono text-zinc-200">qwen3.7-text-embedding</span>, 1024 dims, cosine distance) with exact keyword matching via <span className="font-mono text-zinc-200">BM25</span> (BM25Okapi). Rankings are merged using Reciprocal Rank Fusion (<span className="font-mono text-zinc-200">RRF k=60</span>).
            </p>
            <div className="pt-1.5 flex flex-wrap gap-2 text-[11px] font-mono text-zinc-400">
              <span className="px-2 py-0.5 rounded bg-zinc-800/60 border border-zinc-700/60">Dense: Top-5</span>
              <span className="px-2 py-0.5 rounded bg-zinc-800/60 border border-zinc-700/60">Sparse: Top-5</span>
              <span className="px-2 py-0.5 rounded bg-zinc-800/60 border border-zinc-700/60">Fused Top-K: 5</span>
              <span className="px-2 py-0.5 rounded bg-zinc-800/60 border border-zinc-700/60">Chunk Size: 500-600 tokens</span>
            </div>
          </div>

          {/* Subsystem 2: Verifiable Grounding & Citation Attribution */}
          <div className="p-3.5 rounded-lg bg-[#111114] border border-zinc-800/90 space-y-2">
            <div className="flex items-center space-x-2 text-zinc-200 font-medium">
              <ShieldCheck className="w-3.5 h-3.5 text-indigo-400" />
              <span>Verifiable Grounding &amp; Attribution</span>
            </div>
            <p className="text-zinc-400 leading-relaxed">
              Source-prefixed chunk framing format prevents hallucination across documents. Responses enforce deterministic bracketed citations (<span className="font-mono text-zinc-200">[1]</span>, <span className="font-mono text-zinc-200">[2]</span>) mapped to verifiable chunk metadata with exact page numbers, section titles, and source quotations.
            </p>
          </div>

          {/* Subsystem 3: Dynamic Multi-Doc Scoping Status */}
          <div className="p-3.5 rounded-lg bg-[#111114] border border-zinc-800/90 space-y-2">
            <div className="flex items-center space-x-2 text-zinc-200 font-medium">
              <Layers className="w-3.5 h-3.5 text-indigo-400" />
              <span>Multi-Doc Scoping Status</span>
            </div>
            <p className="text-zinc-400 leading-relaxed">
              {selectedDocCount > 0 ? (
                <>
                  <span className="font-medium text-indigo-300">{selectedDocCount} documents active in scope</span>. Retrieval candidates are isolated with strict metadata filters to guarantee 0% context leakage from unselected documents.
                </>
              ) : (
                <>
                  <span className="font-medium text-zinc-200">Global library search active across all documents</span>. All indexed documents are evaluated in candidate generation.
                </>
              )}
            </p>
          </div>

          {/* Subsystem 4: Keyboard Shortcuts Reference */}
          <div className="p-3.5 rounded-lg bg-[#111114] border border-zinc-800/90 space-y-2.5">
            <div className="flex items-center space-x-2 text-zinc-200 font-medium">
              <Keyboard className="w-3.5 h-3.5 text-indigo-400" />
              <span>Keyboard Shortcuts Reference</span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-[11px] font-mono text-zinc-300">
              <div className="flex items-center justify-between p-1.5 rounded bg-zinc-800/40 border border-zinc-800">
                <span className="text-zinc-400 font-sans">Send Message</span>
                <kbd className="px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-200 border border-zinc-700/60">Ctrl+Enter</kbd>
              </div>
              <div className="flex items-center justify-between p-1.5 rounded bg-zinc-800/40 border border-zinc-800">
                <span className="text-zinc-400 font-sans">Submit (Single line)</span>
                <kbd className="px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-200 border border-zinc-700/60">Enter</kbd>
              </div>
              <div className="flex items-center justify-between p-1.5 rounded bg-zinc-800/40 border border-zinc-800">
                <span className="text-zinc-400 font-sans">Insert Newline</span>
                <kbd className="px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-200 border border-zinc-700/60">Shift+Enter</kbd>
              </div>
              <div className="flex items-center justify-between p-1.5 rounded bg-zinc-800/40 border border-zinc-800">
                <span className="text-zinc-400 font-sans">New Conversation</span>
                <kbd className="px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-200 border border-zinc-700/60">Ctrl+N</kbd>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-zinc-800 bg-[#16161a] flex items-center justify-between text-xs text-zinc-500 font-mono">
          <span>Engine: Qwen 3.8 Flash + ChromaDB</span>
          <button
            type="button"
            onClick={onClose}
            className="px-3 py-1.5 rounded-md bg-zinc-800 hover:bg-zinc-700 text-zinc-200 transition font-sans text-xs cursor-pointer"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
