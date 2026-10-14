import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { DocumentList } from './DocumentList';
import { DocumentItem } from '../../types';

describe('DocumentList Component', () => {
  const mockDocs: DocumentItem[] = [
    {
      id: 'doc_1',
      filename: 'attention_is_all_you_need.pdf',
      pageCount: 15,
      chunkCount: 38,
      uploadedAt: '2026-10-01T12:00:00Z',
      status: 'ready',
    },
    {
      id: 'doc_2',
      filename: 'transformer_xl.pdf',
      pageCount: 20,
      chunkCount: 52,
      uploadedAt: '2026-10-01T14:00:00Z',
      status: 'ready',
    },
  ];

  it('renders document cards with filename, page count, and chunk count metrics', () => {
    render(
      <DocumentList
        documents={mockDocs}
        selectedDocIds={[]}
        isLoading={false}
        onToggleSelect={vi.fn()}
        onSelectAll={vi.fn()}
        onClearSelection={vi.fn()}
        onDeleteDocument={vi.fn()}
        onOpenUpload={vi.fn()}
      />
    );

    expect(screen.getByText('attention_is_all_you_need.pdf')).toBeInTheDocument();
    expect(screen.getByText('transformer_xl.pdf')).toBeInTheDocument();
    expect(screen.getByText(/p\. 15/i)).toBeInTheDocument();
    expect(screen.getByText(/38 chk/i)).toBeInTheDocument();
    expect(screen.getByText(/p\. 20/i)).toBeInTheDocument();
    expect(screen.getByText(/52 chk/i)).toBeInTheDocument();
  });

  it('renders scope indicator correctly when no documents are selected vs scoped', () => {
    const { rerender } = render(
      <DocumentList
        documents={mockDocs}
        selectedDocIds={[]}
        isLoading={false}
        onToggleSelect={vi.fn()}
        onSelectAll={vi.fn()}
        onClearSelection={vi.fn()}
        onDeleteDocument={vi.fn()}
        onOpenUpload={vi.fn()}
      />
    );

    expect(screen.getByText(/Scope: All Documents/i)).toBeInTheDocument();

    rerender(
      <DocumentList
        documents={mockDocs}
        selectedDocIds={['doc_1']}
        isLoading={false}
        onToggleSelect={vi.fn()}
        onSelectAll={vi.fn()}
        onClearSelection={vi.fn()}
        onDeleteDocument={vi.fn()}
        onOpenUpload={vi.fn()}
      />
    );

    expect(screen.getByText(/Scoped: 1 of 2 doc\(s\)/i)).toBeInTheDocument();
  });

  it('calls onToggleSelect when a document card is clicked', () => {
    const handleToggle = vi.fn();
    render(
      <DocumentList
        documents={mockDocs}
        selectedDocIds={[]}
        isLoading={false}
        onToggleSelect={handleToggle}
        onSelectAll={vi.fn()}
        onClearSelection={vi.fn()}
        onDeleteDocument={vi.fn()}
        onOpenUpload={vi.fn()}
      />
    );

    fireEvent.click(screen.getByText('attention_is_all_you_need.pdf'));
    expect(handleToggle).toHaveBeenCalledWith('doc_1');
  });

  it('calls onSelectAll and onClearSelection when master controls are clicked', () => {
    const handleSelectAll = vi.fn();
    const handleClear = vi.fn();

    render(
      <DocumentList
        documents={mockDocs}
        selectedDocIds={['doc_1']}
        isLoading={false}
        onToggleSelect={vi.fn()}
        onSelectAll={handleSelectAll}
        onClearSelection={handleClear}
        onDeleteDocument={vi.fn()}
        onOpenUpload={vi.fn()}
      />
    );

    const selectAllBtn = screen.getByRole('button', { name: /select all/i });
    const clearBtn = screen.getByRole('button', { name: /clear/i });

    fireEvent.click(selectAllBtn);
    expect(handleSelectAll).toHaveBeenCalledTimes(1);

    fireEvent.click(clearBtn);
    expect(handleClear).toHaveBeenCalledTimes(1);
  });

  it('calls onDeleteDocument when the delete button on a document card is clicked', () => {
    const handleDelete = vi.fn();
    render(
      <DocumentList
        documents={mockDocs}
        selectedDocIds={[]}
        isLoading={false}
        onToggleSelect={vi.fn()}
        onSelectAll={vi.fn()}
        onClearSelection={vi.fn()}
        onDeleteDocument={handleDelete}
        onOpenUpload={vi.fn()}
      />
    );

    const deleteButtons = screen.getAllByRole('button', { name: /delete document/i });
    expect(deleteButtons.length).toBe(2);

    fireEvent.click(deleteButtons[0]);
    expect(handleDelete).toHaveBeenCalledWith('doc_1');
  });

  it('renders an empty state with an upload action button when documents array is empty', () => {
    const handleOpenUpload = vi.fn();
    render(
      <DocumentList
        documents={[]}
        selectedDocIds={[]}
        isLoading={false}
        onToggleSelect={vi.fn()}
        onSelectAll={vi.fn()}
        onClearSelection={vi.fn()}
        onDeleteDocument={vi.fn()}
        onOpenUpload={handleOpenUpload}
      />
    );

    expect(screen.getByText(/no documents indexed/i)).toBeInTheDocument();
    const uploadBtns = screen.getAllByRole('button', { name: /upload pdf/i });
    expect(uploadBtns.length).toBeGreaterThanOrEqual(1);
    fireEvent.click(uploadBtns[uploadBtns.length - 1]);
    expect(handleOpenUpload).toHaveBeenCalledTimes(1);
  });

  it('renders skeleton loaders or loading state when isLoading is true', () => {
    render(
      <DocumentList
        documents={[]}
        selectedDocIds={[]}
        isLoading={true}
        onToggleSelect={vi.fn()}
        onSelectAll={vi.fn()}
        onClearSelection={vi.fn()}
        onDeleteDocument={vi.fn()}
        onOpenUpload={vi.fn()}
      />
    );

    expect(screen.getByTestId('document-list-skeleton')).toBeInTheDocument();
  });

  it('toggles section collapse when accordion header is clicked', () => {
    render(
      <DocumentList
        documents={mockDocs}
        selectedDocIds={[]}
        isLoading={false}
        onToggleSelect={vi.fn()}
        onSelectAll={vi.fn()}
        onClearSelection={vi.fn()}
        onDeleteDocument={vi.fn()}
        onOpenUpload={vi.fn()}
      />
    );

    expect(screen.getByText('attention_is_all_you_need.pdf')).toBeInTheDocument();

    const headerToggle = screen.getByRole('button', { name: /toggle corpus explorer/i });
    fireEvent.click(headerToggle);

    // When collapsed, document items are hidden
    expect(screen.queryByText('attention_is_all_you_need.pdf')).not.toBeInTheDocument();

    // Clicking again expands the section
    fireEvent.click(headerToggle);
    expect(screen.getByText('attention_is_all_you_need.pdf')).toBeInTheDocument();
  });
});
