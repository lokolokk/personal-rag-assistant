import React, { useEffect, useRef, useState } from 'react';
import App from '../../App';
import { ACCESS_EVENT, OFFLINE_EVENT, AccessSession, apiBase, authenticatedFetch,
  clearSession, readSession, redeemInvitation, verifySession } from '../../services/session';

// Capture once before React starts: StrictMode must not redeem twice.
const fragment = new URLSearchParams(window.location.hash.slice(1));
let pendingInvitation = fragment.get('invite');
if (pendingInvitation) {
  clearSession();
  window.history.replaceState(null, '', window.location.pathname + window.location.search);
}
let redemption: Promise<AccessSession> | null = null;

export default function AccessGate(): React.JSX.Element {
  const [session, setSession] = useState<AccessSession | null>(null);
  const [status, setStatus] = useState('Checking access…');
  const [offline, setOffline] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const revision = useRef(0);
  useEffect(() => {
    let current = true;
    const accessChanged = (): void => {
      if (!readSession()) { revision.current += 1; setSession(null); setStatus('Access expired or revoked. Reopen a valid invitation link.'); }
    };
    const unavailable = (): void => { setOffline(true); setSession(null); setStatus('Backend unavailable. The host’s laptop or connection may be offline.'); };
    window.addEventListener(ACCESS_EVENT, accessChanged);
    window.addEventListener(OFFLINE_EVENT, unavailable);
    setOffline(false);
    const action = pendingInvitation
      ? (redemption ??= redeemInvitation(pendingInvitation))
      : readSession() ? verifySession() : Promise.reject(new Error('Open your invitation link to access your workspace.'));
    void action.then(value => {
      pendingInvitation = null;
      redemption = null;
      if (current) { setSession(value); setStatus(''); setOffline(false); }
    }).catch((error: unknown) => {
      redemption = null;
      if (current) {
        setSession(null);
        setStatus(error instanceof Error && error.name === 'AccessError' ? error.message
          : pendingInvitation || readSession() ? 'Backend unavailable. Check the host connection and retry.'
          : 'Open your invitation link to access your workspace.');
      }
    });
    return () => { current = false; window.removeEventListener(ACCESS_EVENT, accessChanged); window.removeEventListener(OFFLINE_EVENT, unavailable); };
  }, [attempt]);
  useEffect(() => {
    if (!session) return;
    const timer = window.setTimeout(() => clearSession(), Math.max(0, Date.parse(session.expiresAt)-Date.now()));
    return () => window.clearTimeout(timer);
  }, [session]);
  async function leave(): Promise<void> {
    try { await authenticatedFetch(`${apiBase()}/access/session`, { method: 'DELETE' }); }
    finally { clearSession(); }
  }
  if (session && !offline) return <>
    <div className="bg-zinc-950 text-zinc-300 text-xs px-4 py-2 flex justify-between">
      <span>Shared workspace · Anyone with this invitation shares its files and chats.</span>
      <button type="button" onClick={() => { void leave().catch(() => setStatus('Local access cleared.')); }}>Leave workspace</button>
    </div>
    <App key={`${session.workspaceId}:${revision.current}`} />
  </>;
  return <main className="min-h-screen bg-zinc-950 text-zinc-100 flex items-center justify-center p-6">
    <div className="max-w-md space-y-4"><h1 className="text-xl font-semibold">Personal RAG Assistant</h1>
      <p role="status">{status}</p>
      <button type="button" className="border border-zinc-700 rounded px-4 py-2" onClick={() => setAttempt(value => value+1)}>Retry connection</button>
    </div>
  </main>;
}
