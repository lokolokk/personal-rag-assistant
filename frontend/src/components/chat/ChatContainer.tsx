import React, { useState, useRef, useEffect } from 'react';
import {
  Send,
  Square,
  Layers,
  ArrowRight,
} from 'lucide-react';
import { DisplayMessage, CitationDetail } from '../../types';
import { MessageBubble } from './MessageBubble';

export interface ChatContainerProps {
  messages: DisplayMessage[];
  isStreaming: boolean;
  selectedDocCount: number;
  onSendMessage: (query: string) => void | Promise<void>;
  onStopStream: () => void;
  onSelectCitation?: (citation: CitationDetail) => void;
  onOpenSpecs?: () => void;
  className?: string;
}

export function ChatContainer({
  messages,
  isStreaming,
  selectedDocCount,
  onSendMessage,
  onStopStream,
  onSelectCitation,
  onOpenSpecs,
  className = '',
}: ChatContainerProps): React.JSX.Element {
  const [inputValue, setInputValue] = useState<string>('');
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const isUserScrolledUp = useRef<boolean>(false);

  // Monitor user scrolling to manage auto-scroll lock
  const handleScroll = (): void => {
    if (!scrollContainerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollContainerRef.current;
    // User is considered scrolled up if they are > 80px from bottom
    isUserScrolledUp.current = scrollHeight - scrollTop - clientHeight > 80;
  };

  // Auto-scroll on new messages or tokens unless user explicitly scrolled up
  useEffect(() => {
    if (
      !isUserScrolledUp.current &&
      messagesEndRef.current &&
      typeof messagesEndRef.current.scrollIntoView === 'function'
    ) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // Adjust textarea height dynamically based on input length
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const scrollHeight = textareaRef.current.scrollHeight;
      textareaRef.current.style.height = `${Math.min(scrollHeight, 180)}px`;
    }
  }, [inputValue]);

  const handleSubmit = (e?: React.FormEvent): void => {
    if (e) e.preventDefault();
    const query = inputValue.trim();
    if (!query || isStreaming) return;

    onSendMessage(query);
    setInputValue('');
    isUserScrolledUp.current = false;
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>): void => {
    if (e.key === 'Enter') {
      if (e.shiftKey) {
        // Shift+Enter inserts newline
        return;
      }
      // Enter or Ctrl+Enter submits
      e.preventDefault();
      handleSubmit();
    }
  };

  const suggestedPrompts = [
    'Summarize the key methodology and structural findings',
    'Compare the experimental results across model architectures',
    'What are the core limitations identified by the authors?',
  ];

  return (
    <div className={`flex flex-col h-full bg-[#09090b] text-zinc-100 overflow-hidden relative ${className}`}>
      {/* Messages Thread Container */}
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto px-4 sm:px-8 py-6 space-y-4"
      >
        {messages.length === 0 ? (
          /* Serene Distilled Empty State */
          <div
            data-testid="corpus-telemetry-grid"
            className="h-full min-h-[380px] flex flex-col justify-center max-w-2xl mx-auto px-4 py-8"
          >
            {/* Header / Subsystem info */}
            <div className="text-center mb-8">
              <h2 className="text-base sm:text-lg font-semibold text-zinc-100 tracking-tight">
                Conversational Document RAG
              </h2>
              <p className="text-xs text-zinc-400 mt-1.5 font-sans">
                Ask questions, compare findings, and explore your documents.
              </p>
            </div>

            {/* Flat Borderless Prompt Suggestions */}
            <div className="w-full space-y-2 mb-6">
              <div className="flex items-center justify-between text-[11px] font-mono uppercase tracking-wider text-zinc-400 mb-1 px-1 font-semibold">
                <span>Suggested Questions</span>
                <span className="text-[10px] text-zinc-500 font-normal">Click to execute</span>
              </div>
              <div className="flex flex-col space-y-1">
                {suggestedPrompts.map((prompt, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => onSendMessage(prompt)}
                    className="w-full py-2.5 px-3 rounded-lg text-left text-xs font-medium text-zinc-300 hover:text-white hover:bg-zinc-800/60 transition flex items-center justify-between group cursor-pointer"
                  >
                    <div className="flex items-center space-x-2.5 min-w-0">
                      <span className="font-mono text-indigo-400 text-xs group-hover:translate-x-0.5 transition-transform shrink-0">
                        &gt;
                      </span>
                      <span className="truncate">{prompt}</span>
                    </div>
                    <ArrowRight className="w-3.5 h-3.5 text-zinc-600 group-hover:text-indigo-400 group-hover:translate-x-0.5 transition shrink-0 ml-2" />
                  </button>
                ))}
              </div>
            </div>

            {/* Quiet Inline Trigger for Architecture & Telemetry Specs */}
            {onOpenSpecs && (
              <div className="text-center pt-2">
                <button
                  type="button"
                  onClick={onOpenSpecs}
                  className="inline-flex items-center space-x-1.5 text-xs text-zinc-500 hover:text-indigo-400 transition cursor-pointer font-sans"
                  aria-label="View system architecture & telemetry specs"
                >
                  <span>View System Architecture &amp; Telemetry Specs</span>
                  <span>&rarr;</span>
                </button>
              </div>
            )}
          </div>
        ) : (
          <div className="max-w-5xl mx-auto w-full">
            {messages.map((msg) => (
              <MessageBubble
                key={msg.id}
                message={msg}
                onSelectCitation={onSelectCitation}
              />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Docked Edge-to-Edge Command Console */}
      <div
        data-testid="docked-command-console"
        className="border-t border-zinc-800 bg-[#0e0e11] shrink-0"
      >
        <div className="max-w-5xl mx-auto w-full px-4 sm:px-8 py-3 flex flex-col space-y-2">
          {/* Top Bar: Retrieval Scope Indicator & Keyboard Shortcut Cues */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Layers className="w-3.5 h-3.5 text-zinc-400" />
              <span className="text-[11px] font-medium text-zinc-400">Retrieval Scope:</span>
              <span
                className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-mono font-medium ${
                  selectedDocCount > 0
                    ? 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/30'
                    : 'bg-zinc-800/80 text-zinc-400 border border-zinc-700/60'
                }`}
              >
                {selectedDocCount > 0
                  ? `${selectedDocCount} docs scoped`
                  : 'All documents (global)'}
              </span>
            </div>

            <div className="hidden sm:flex items-center space-x-2 text-[11px] font-mono text-zinc-500">
              <span className="flex items-center gap-1">
                <kbd className="px-1.5 py-0.5 rounded bg-zinc-800/80 border border-zinc-700/60 text-zinc-300 text-[10px] font-mono">Ctrl+Enter</kbd>
                <span>to send</span>
              </span>
              <span className="text-zinc-700">•</span>
              <span className="flex items-center gap-1">
                <kbd className="px-1.5 py-0.5 rounded bg-zinc-800/80 border border-zinc-700/60 text-zinc-300 text-[10px] font-mono">Shift+Enter</kbd>
                <span>for newline</span>
              </span>
            </div>
          </div>

          {/* Textarea & Actions Row */}
          <div className="flex items-end gap-2 bg-[#141417] border border-zinc-800 rounded-lg p-2 focus-within:border-indigo-500/60 focus-within:ring-1 focus-within:ring-indigo-500/30 transition-all">
            <textarea
              ref={textareaRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                selectedDocCount > 0
                  ? `Ask questions grounded in ${selectedDocCount} selected document(s)...`
                  : 'Ask questions grounded in uploaded documents...'
              }
              rows={1}
              className="flex-1 max-h-44 p-1.5 text-sm text-zinc-100 placeholder:text-zinc-500 bg-transparent resize-none focus:outline-none leading-relaxed font-sans"
            />

            {/* Action Button: Send or Stop */}
            {isStreaming ? (
              <button
                type="button"
                onClick={onStopStream}
                className="p-2 rounded-md bg-zinc-800 text-white hover:bg-zinc-700 border border-zinc-700 transition shrink-0 shadow-2xs active:scale-95 cursor-pointer"
                title="Stop generating"
                aria-label="Stop generating"
              >
                <Square className="w-4 h-4 fill-white" />
              </button>
            ) : (
              <button
                type="button"
                onClick={() => handleSubmit()}
                disabled={!inputValue.trim()}
                className="p-2 rounded-md bg-indigo-600 text-white font-medium hover:bg-indigo-500 disabled:opacity-30 disabled:cursor-not-allowed transition shrink-0 shadow-xs active:scale-95 cursor-pointer"
                title="Send message"
                aria-label="Send message"
              >
                <Send className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
