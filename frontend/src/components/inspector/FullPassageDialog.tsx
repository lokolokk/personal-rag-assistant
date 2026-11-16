import React, { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { ApiError, fetchFullPassage } from '../../services/api';
import { CitationDetail, FullPassage } from '../../types';

export function FullPassageDialog({ citation, onClose }: {
  citation: CitationDetail;
  onClose: () => void;
}): React.JSX.Element {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const [attempt, setAttempt] = useState(0);
  const [passage, setPassage] = useState<FullPassage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState(false);

  useEffect(() => {
    const trigger = document.activeElement;
    const dialog = dialogRef.current;
    if (dialog?.showModal) dialog.showModal();
    else dialog?.setAttribute('open', '');
    closeRef.current?.focus();
    const overflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = overflow;
      if (dialog?.open) {
        if (dialog.close) dialog.close();
        else dialog.removeAttribute('open');
      }
      if (trigger instanceof HTMLElement && trigger.isConnected) trigger.focus({ preventScroll: true });
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    let current = true;
    setPassage(null);
    setError(null);
    setCopied(false);
    setCopyError(false);
    void fetchFullPassage(citation.chunkId, controller.signal).then(result => {
      if (current) setPassage(result);
    }).catch((reason: unknown) => {
      if (!current) return;
      setError(reason instanceof ApiError && reason.status === 404
        ? 'This source is unavailable or has been deleted.'
        : 'Could not load the full passage.');
    });
    return () => { current = false; controller.abort(); };
  }, [citation.chunkId, attempt]);

  const copy = async (): Promise<void> => {
    if (!passage) return;
    try {
      await navigator.clipboard.writeText(passage.text);
      setCopied(true);
      setCopyError(false);
    } catch { setCopyError(true); }
  };

  return createPortal(
    <dialog ref={dialogRef} aria-label="Full passage" aria-modal="true"
      className="fixed m-auto w-[min(56rem,calc(100%-2rem))] max-h-[85vh] rounded-lg border border-zinc-700 bg-zinc-950 text-zinc-100 p-0 backdrop:bg-black/70"
      onCancel={event => { event.preventDefault(); onClose(); }}
      onKeyDown={event => {
        if (event.key === 'Escape') {
          event.preventDefault(); event.stopPropagation(); onClose();
        }
        if (event.key === 'Tab') {
          const controls = Array.from(event.currentTarget.querySelectorAll<HTMLElement>(
            'button:not(:disabled), [tabindex="0"]',
          ));
          const first = controls[0], last = controls[controls.length - 1];
          if (event.shiftKey && document.activeElement === first) {
            event.preventDefault(); last?.focus();
          } else if (!event.shiftKey && document.activeElement === last) {
            event.preventDefault(); first?.focus();
          }
        }
      }}>
      <div className="flex items-center justify-between p-4 border-b border-zinc-800">
        <h2 className="font-semibold">Full passage</h2>
        <button ref={closeRef} type="button" onClick={onClose} aria-label="Close full passage"
          className="rounded px-3 py-1 bg-zinc-800 hover:bg-zinc-700">Close</button>
      </div>
      <div className="p-4 space-y-4">
        {!passage && !error && <p role="status">Loading full passage…</p>}
        {error && <div role="alert"><p>{error}</p>
          <button type="button" onClick={() => setAttempt(value => value + 1)}
            className="mt-2 rounded px-3 py-1 bg-zinc-800">Retry</button></div>}
        {passage && <>
          <p className="text-sm break-words">{passage.filename} • Page {passage.pageNumber}</p>
          <p className="font-mono text-xs break-all">{passage.chunkId}</p>
          <div tabIndex={0} role="region" aria-label="Full source text"
            className="max-h-[50vh] overflow-y-auto overscroll-contain rounded border border-zinc-800 p-4">
            <pre className="whitespace-pre-wrap break-words font-mono text-sm leading-relaxed">{passage.text}</pre>
          </div>
          <button type="button" onClick={() => void copy()} aria-label="Copy full passage"
            className="rounded px-3 py-2 bg-zinc-800 hover:bg-zinc-700">
            {copied ? 'Copied' : 'Copy full passage'}</button>
          {copyError && <p role="alert">Could not copy. Select the passage and copy it manually.</p>}
        </>}
      </div>
    </dialog>, document.body,
  );
}
