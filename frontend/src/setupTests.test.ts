import { describe, it, expect } from 'vitest';

describe('offline test default', () => {
  it('rejects an unmocked fetch before network dispatch', async () => {
    await expect(fetch('/offline-check')).rejects.toThrow('Unmocked fetch is disabled');
  });
});
