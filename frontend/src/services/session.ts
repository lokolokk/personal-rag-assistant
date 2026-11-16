/** Short-lived bearer access; invitations remain only in memory. */
export interface AccessSession {
  sessionToken: string;
  workspaceId: string;
  expiresAt: string;
  limits: Record<string, number>;
}

const STORAGE_KEY = 'rag-session-v1';
export const ACCESS_EVENT = 'rag-access-change';
export const OFFLINE_EVENT = 'rag-backend-unavailable';
export function invitedAccess(): boolean { return import.meta.env.VITE_INVITED_ACCESS === '1'; }
export function apiBase(): string {
  return (import.meta.env.VITE_API_URL as string | undefined) ?? '/api/v1';
}

export function clearSession(): void {
  sessionStorage.removeItem(STORAGE_KEY);
  window.dispatchEvent(new Event(ACCESS_EVENT));
}

export function readSession(): AccessSession | null {
  try {
    const raw: unknown = JSON.parse(sessionStorage.getItem(STORAGE_KEY) ?? 'null');
    if (typeof raw !== 'object' || raw === null || !('sessionToken' in raw)
      || !('workspaceId' in raw) || !('expiresAt' in raw) || !('limits' in raw)) return null;
    if (typeof raw.sessionToken !== 'string' || typeof raw.workspaceId !== 'string'
      || typeof raw.expiresAt !== 'string' || !Number.isFinite(Date.parse(raw.expiresAt))
      || Date.parse(raw.expiresAt) <= Date.now()) return null;
    return raw as AccessSession;
  } catch { return null; }
}

export class AccessError extends Error {
  constructor(public code: string, message: string) { super(message); this.name = 'AccessError'; }
}

function checkedUrl(url: string): string {
  const base = new URL(apiBase(), window.location.origin);
  const target = new URL(url, window.location.origin);
  const local = ['localhost', '127.0.0.1', '[::1]'].includes(base.hostname);
  if ((base.protocol !== 'https:' && !local) || target.origin !== base.origin
    || !target.pathname.startsWith(`${base.pathname}/`) || target.search || target.hash
    || target.username || target.password) throw new AccessError('CONFIGURATION', 'Backend address is invalid.');
  return target.href;
}

export async function authenticatedFetch(url: string, options: RequestInit = {}, redeem = false, reportOffline = true): Promise<Response> {
  if (!invitedAccess()) return fetch(url, options);
  const destination = checkedUrl(url);
  const session = readSession();
  if (!redeem && !session) throw new AccessError('ACCESS_EXPIRED', 'Open your invitation link to continue.');
  const headers = new Headers(options.headers);
  if (session && !redeem) headers.set('Authorization', `Bearer ${session.sessionToken}`);
  if (import.meta.env.VITE_NGROK_BYPASS === '1') headers.set('ngrok-skip-browser-warning', '1');
  let response: Response;
  const controller = new AbortController();
  const abort = (): void => controller.abort();
  options.signal?.addEventListener('abort', abort, { once: true });
  if (options.signal?.aborted) controller.abort();
  const timer = window.setTimeout(abort, options.body instanceof FormData ? 190000 : 15000);
  try {
    response = await fetch(destination, { ...options, signal: controller.signal, headers, credentials: 'omit', cache: 'no-store', redirect: 'error' });
  } catch (error) {
    if (reportOffline && !options.signal?.aborted) window.dispatchEvent(new Event(OFFLINE_EVENT));
    throw error;
  } finally {
    window.clearTimeout(timer);
  }
  if (response.status === 401) {
    const data: unknown = await response.clone().json().catch(() => null);
    const code = typeof data === 'object' && data !== null && 'error' in data
      && typeof data.error === 'object' && data.error !== null && 'code' in data.error
      ? String(data.error.code) : 'ACCESS_INVALID';
    if (!redeem) clearSession();
    const message = code === 'ACCESS_REVOKED' ? 'This invitation has been revoked. Ask the host for a new link.'
      : code === 'ACCESS_EXPIRED' ? 'Access has expired. Reopen your invitation or ask the host for a new link.'
      : 'Invalid access. Open a valid invitation link.';
    throw new AccessError(code, message);
  }
  if (!redeem && session?.sessionToken !== readSession()?.sessionToken) {
    throw new AccessError('ACCESS_CHANGED', 'Workspace access changed.');
  }
  return response;
}

export async function redeemInvitation(invitation: string): Promise<AccessSession> {
  const response = await authenticatedFetch(`${apiBase()}/access/redeem`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ invitation }),
  }, true);
  if (!response.ok) throw new AccessError('ACCESS_INVALID', 'Invitation could not be redeemed.');
  const data = await response.json() as AccessSession;
  sessionStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  if (!readSession()) { clearSession(); throw new AccessError('ACCESS_INVALID', 'Invalid session response.'); }
  window.dispatchEvent(new Event(ACCESS_EVENT));
  return data;
}

export async function verifySession(): Promise<AccessSession> {
  const response = await authenticatedFetch(`${apiBase()}/access/session`);
  if (!response.ok) throw new Error('Backend is unavailable.');
  const data = await response.json() as Omit<AccessSession, 'sessionToken'>;
  const old = readSession();
  if (!old) throw new AccessError('ACCESS_EXPIRED', 'Session expired.');
  const value = { ...data, sessionToken: old.sessionToken };
  sessionStorage.setItem(STORAGE_KEY, JSON.stringify(value));
  return value;
}
