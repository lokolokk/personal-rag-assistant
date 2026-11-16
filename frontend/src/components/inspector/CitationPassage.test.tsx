import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { CitationDrawer } from './CitationDrawer';
import * as api from '../../services/api';
import { CitationDetail } from '../../types';
import { StrictMode } from 'react';

const citation: CitationDetail = {
  citationIndex: 1, chunkId: 'chunk_a', documentId: 'doc_a', filename: 'a.pdf',
  pageNumber: 7, sectionTitle: '', sourceText: 'x'.repeat(200) + '...',
};
const passage = { chunkId: 'chunk_a', filename: 'a.pdf', pageNumber: 7, text: '<script>literal</script> full source' };

describe('Citation passage', () => {
  beforeEach(() => { vi.restoreAllMocks(); });

  it('keeps preview separate, fetches on demand, and copies the requested text', async () => {
    const user = userEvent.setup();
    const copy = vi.spyOn(navigator.clipboard, 'writeText').mockResolvedValue();
    const fetch = vi.spyOn(api, 'fetchFullPassage').mockResolvedValue(passage);
    render(<CitationDrawer isOpen citation={{ ...citation, isTruncated: true }} onClose={vi.fn()} />);
    expect(screen.getByText('Source preview — shortened')).toBeInTheDocument();
    expect(fetch).not.toHaveBeenCalled();
    await user.click(screen.getByRole('button', { name: 'Copy preview' }));
    expect(copy).toHaveBeenLastCalledWith(citation.sourceText);
    await user.click(screen.getByRole('button', { name: 'View full passage' }));
    const dialog = await screen.findByRole('dialog', { name: 'Full passage' });
    expect(await within(dialog).findByText(passage.text)).toBeInTheDocument();
    expect(dialog.querySelector('script')).toBeNull();
    expect(within(dialog).getByText('chunk_a')).toBeInTheDocument();
    expect(within(dialog).getByText(/Page 7/)).toBeInTheDocument();
    await user.click(within(dialog).getByRole('button', { name: 'Copy full passage' }));
    expect(copy).toHaveBeenLastCalledWith(passage.text);
    expect(screen.getByText(citation.sourceText)).toBeInTheDocument();
  });

  it('traps focus, closes only the popup on Escape, restores focus and reopens', async () => {
    const user = userEvent.setup();
    vi.spyOn(api, 'fetchFullPassage').mockResolvedValue(passage);
    const closeInspector = vi.fn();
    render(<StrictMode><CitationDrawer isOpen citation={citation} onClose={closeInspector} /></StrictMode>);
    const trigger = screen.getByRole('button', { name: 'View full passage' });
    await user.click(trigger);
    const dialog = await screen.findByRole('dialog', { name: 'Full passage' });
    const close = within(dialog).getByRole('button', { name: 'Close full passage' });
    expect(close).toHaveFocus();
    await user.tab({ shift: true });
    expect(dialog.contains(document.activeElement)).toBe(true);
    await user.tab();
    expect(close).toHaveFocus();
    await user.keyboard('{Escape}');
    expect(screen.queryByRole('dialog', { name: 'Full passage' })).not.toBeInTheDocument();
    expect(closeInspector).not.toHaveBeenCalled();
    expect(trigger).toHaveFocus();
    await user.click(trigger);
    await user.click(screen.getByRole('button', { name: 'Close full passage' }));
    expect(trigger).toHaveFocus();
  });

  it('shows loading, retryable errors, and unavailable sources', async () => {
    const fetch = vi.spyOn(api, 'fetchFullPassage').mockRejectedValueOnce(new Error('network'))
      .mockRejectedValueOnce(new api.ApiError('missing', 404));
    render(<CitationDrawer isOpen citation={citation} onClose={vi.fn()} />);
    fireEvent.click(screen.getByRole('button', { name: 'View full passage' }));
    expect(screen.getByText('Loading full passage…')).toBeInTheDocument();
    expect(await screen.findByText('Could not load the full passage.')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(await screen.findByText('This source is unavailable or has been deleted.')).toBeInTheDocument();
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(screen.queryByRole('button', { name: 'Copy full passage' })).not.toBeInTheDocument();
  });

  it('discards late results after switching citations or closing', async () => {
    let resolve!: (value: typeof passage) => void;
    vi.spyOn(api, 'fetchFullPassage').mockImplementationOnce(() => new Promise(r => { resolve = r; }))
      .mockResolvedValue({ ...passage, chunkId: 'chunk_b', text: 'Second full passage' });
    const props = { isOpen: true, citation, onClose: vi.fn() };
    const { rerender } = render(<CitationDrawer {...props} />);
    fireEvent.click(screen.getByRole('button', { name: 'View full passage' }));
    rerender(<CitationDrawer {...props} citation={{ ...citation, chunkId: 'chunk_b' }} />);
    expect(screen.queryByRole('dialog', { name: 'Full passage' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'View full passage' }));
    expect(await screen.findByText('Second full passage')).toBeInTheDocument();
    await act(async () => { resolve(passage); });
    expect(screen.queryByText(passage.text)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Close full passage' }));
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Full passage' })).not.toBeInTheDocument());
  });

  it('does not infer shortening from a natural ellipsis', () => {
    render(<CitationDrawer isOpen citation={{ ...citation, isTruncated: false }} onClose={vi.fn()} />);
    expect(screen.getByText('Source preview')).toBeInTheDocument();
    expect(screen.queryByText('Source preview — shortened')).not.toBeInTheDocument();
  });

  it('ignores a pending response after closing and reopening the dialog', async () => {
    let resolve!: (value: typeof passage) => void;
    vi.spyOn(api, 'fetchFullPassage').mockImplementationOnce(() => new Promise(r => { resolve = r; }))
      .mockResolvedValue({ ...passage, text: 'Fresh result' });
    render(<CitationDrawer isOpen citation={citation} onClose={vi.fn()} />);
    fireEvent.click(screen.getByRole('button', { name: 'View full passage' }));
    fireEvent.click(screen.getByRole('button', { name: 'Close full passage' }));
    fireEvent.click(screen.getByRole('button', { name: 'View full passage' }));
    expect(await screen.findByText('Fresh result')).toBeInTheDocument();
    await act(async () => { resolve(passage); });
    expect(screen.queryByText(passage.text)).not.toBeInTheDocument();
  });
});
