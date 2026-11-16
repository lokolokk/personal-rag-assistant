import { afterEach, beforeEach, expect, it, vi } from 'vitest';

beforeEach(() => {
  vi.resetModules();
  vi.stubEnv('VITE_INVITED_ACCESS', '1');
  vi.stubEnv('VITE_API_URL', 'https://backend.example/api/v1');
  sessionStorage.setItem('rag-session-v1', JSON.stringify({
    sessionToken: 'test-session', workspaceId: 'a', expiresAt: '2099-01-01T00:00:00Z', limits: {},
  }));
});
afterEach(() => { vi.unstubAllEnvs(); vi.unstubAllGlobals(); sessionStorage.clear(); });

it('fetches a passage with bearer authentication and rejects mismatched source IDs', async () => {
  const fetch = vi.fn().mockImplementation(() => Promise.resolve(new Response(JSON.stringify({
    chunkId: 'source', text: 'Full original', filename: 'source.pdf', pageNumber: 7,
  }))));
  vi.stubGlobal('fetch', fetch);
  const { fetchFullPassage } = await import('./api');
  expect((await fetchFullPassage('source')).text).toBe('Full original');
  expect(fetch.mock.calls[0][0]).toBe('https://backend.example/api/v1/chunks/source');
  expect(new Headers(fetch.mock.calls[0][1].headers).get('Authorization')).toBe('Bearer test-session');
  await expect(fetchFullPassage('other')).rejects.toThrow('Invalid passage response');
});

it('keeps passage network failures local to the dialog for retry', async () => {
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline')));
  const { fetchFullPassage } = await import('./api');
  const { OFFLINE_EVENT } = await import('./session');
  const offline = vi.fn();
  window.addEventListener(OFFLINE_EVENT, offline);
  try {
    await expect(fetchFullPassage('source')).rejects.toThrow('offline');
    expect(offline).not.toHaveBeenCalled();
  } finally { window.removeEventListener(OFFLINE_EVENT, offline); }
});
