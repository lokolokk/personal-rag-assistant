import { beforeEach, describe, expect, it, vi } from 'vitest';
import { clearSession, redeemInvitation, authenticatedFetch, readSession } from './session';

describe('invited session handling', () => {
  beforeEach(() => {
    sessionStorage.clear();
    vi.stubEnv('VITE_INVITED_ACCESS', '1');
    vi.stubEnv('VITE_API_URL', 'https://backend.example/api/v1');
    clearSession();
  });

  it('redeems through the fixed origin and stores no invitation', async () => {
    const invitation = 'test-invitation';
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      sessionToken: 'test-session', workspaceId: 'a', expiresAt: '2099-01-01T00:00:00Z', limits: {},
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }));
    vi.stubGlobal('fetch', fetchMock);
    await redeemInvitation(invitation);
    expect(fetchMock.mock.calls[0][0]).toBe('https://backend.example/api/v1/access/redeem');
    expect(JSON.stringify(sessionStorage)).not.toContain(invitation);
    expect(readSession()?.workspaceId).toBe('a');
  });

  it('authenticates uploads without assigning a JSON content type', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      sessionToken: 'test-session', workspaceId: 'a', expiresAt: '2099-01-01T00:00:00Z', limits: {},
    }), { status: 200 })));
    await redeemInvitation('test-invitation');
    const body = new FormData();
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}'));
    vi.stubGlobal('fetch', fetchMock);
    await authenticatedFetch('https://backend.example/api/v1/documents/upload', { method: 'POST', body });
    const options = fetchMock.mock.calls[0][1] as RequestInit;
    expect(new Headers(options.headers).get('Authorization')).toBe('Bearer test-session');
    expect(new Headers(options.headers).has('Content-Type')).toBe(false);
    expect(options.credentials).toBe('omit');
  });

  it('rejects a destination outside configured API', async () => {
    const spy = vi.fn();
    vi.stubGlobal('fetch', spy);
    await expect(authenticatedFetch('https://foreign.example/api/v1/documents')).rejects.toThrow();
    expect(spy).not.toHaveBeenCalled();
  });

  it('keeps stream cancellation connected after response headers arrive', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      sessionToken:'test-session',workspaceId:'a',expiresAt:'2099-01-01T00:00:00Z',limits:{},
    }))));
    await redeemInvitation('test-invitation');
    const spy = vi.fn().mockResolvedValue(new Response('event: status\ndata: {}\n\n'));
    vi.stubGlobal('fetch',spy);
    const controller = new AbortController();
    await authenticatedFetch('https://backend.example/api/v1/conversations/a/messages', {signal:controller.signal});
    controller.abort();
    expect((spy.mock.calls[0][1] as RequestInit).signal?.aborted).toBe(true);
  });
});
