import { beforeEach, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, within } from '@testing-library/react';
import App from './App';
import * as api from './services/api';

beforeEach(() => { vi.restoreAllMocks(); });

it('keeps repeated citation numbers and shared chunks owned by the clicked answer', async () => {
  vi.spyOn(api, 'fetchDocuments').mockResolvedValue([]);
  vi.spyOn(api, 'fetchConversations').mockResolvedValue([{
    id: 'conv_test', title: 'Citation ownership', createdAt: '', updatedAt: '', documentIds: [], messageCount: 2,
  }]);
  const citation = (id: number, chunk: string, filename: string) => ({
    id, chunk_id: chunk, document_id: 'doc', filename, page_number: 1, quote: `Preview ${filename}`, isTruncated: true,
  });
  vi.spyOn(api, 'fetchConversationMessages').mockResolvedValue([
    { role: 'assistant', content: 'Earlier [1] [2]', citations: [citation(1, 'shared', 'shared.pdf'), citation(2, 'old', 'old.pdf')] },
    { role: 'assistant', content: 'Later [1] [2]', citations: [citation(1, 'shared', 'shared.pdf'), citation(2, 'new', 'new.pdf')] },
  ]);
  render(<App />);
  fireEvent.click(await screen.findByRole('button', { name: /Citation ownership/i }));
  const badges = await screen.findAllByRole('button', { name: /Citation 1/ });
  fireEvent.click(badges[1]);
  const drawer = screen.getByRole('dialog', { name: 'Citation Inspector' });
  expect(within(drawer).getByRole('button', { name: /Citation 2: new.pdf/ })).toBeInTheDocument();
  expect(within(drawer).queryByRole('button', { name: /old.pdf/ })).not.toBeInTheDocument();
  expect(within(drawer).getByText('Source preview — shortened')).toBeInTheDocument();
  fireEvent.click(within(drawer).getByRole('button', { name: 'Next citation' }));
  expect(within(drawer).getByText('Preview new.pdf')).toBeInTheDocument();
  fireEvent.click(screen.getAllByRole('button', { name: /Citation 1/ })[0]);
  expect(within(screen.getByRole('dialog', { name: 'Citation Inspector' }))
    .getByRole('button', { name: /Citation 2: old.pdf/ })).toBeInTheDocument();
});
