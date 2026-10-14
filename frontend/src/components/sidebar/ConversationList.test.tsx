import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ConversationList } from './ConversationList';
import { ConversationItem } from '../../types';

describe('ConversationList Component', () => {
  const mockConversations: ConversationItem[] = [
    {
      id: 'conv_1',
      title: 'Multi-Head Attention Query',
      createdAt: '2026-10-02T10:00:00Z',
      updatedAt: new Date(Date.now() - 5 * 60 * 1000).toISOString(), // 5 mins ago
      documentIds: ['doc_1'],
      messageCount: 4,
    },
    {
      id: 'conv_2',
      title: 'Segment-Level Recurrence in XL',
      createdAt: '2026-10-01T12:00:00Z',
      updatedAt: '2026-10-01T12:30:00Z',
      documentIds: ['doc_1', 'doc_2'],
      messageCount: 6,
    },
  ];

  it('renders empty state when conversation list is empty', () => {
    render(
      <ConversationList
        conversations={[]}
        activeConversationId={null}
        onSelectConversation={vi.fn()}
        onDeleteConversation={vi.fn()}
        onNewChat={vi.fn()}
      />
    );

    expect(screen.getByText(/no past conversations/i)).toBeInTheDocument();
    expect(
      screen.getByText(/start asking questions to build your history/i)
    ).toBeInTheDocument();
  });

  it('renders conversation items with titles and relative time', () => {
    render(
      <ConversationList
        conversations={mockConversations}
        activeConversationId="conv_1"
        onSelectConversation={vi.fn()}
        onDeleteConversation={vi.fn()}
        onNewChat={vi.fn()}
      />
    );

    expect(screen.getByText('Multi-Head Attention Query')).toBeInTheDocument();
    expect(screen.getByText('Segment-Level Recurrence in XL')).toBeInTheDocument();
  });

  it('highlights active conversation item with aria-current="true"', () => {
    render(
      <ConversationList
        conversations={mockConversations}
        activeConversationId="conv_1"
        onSelectConversation={vi.fn()}
        onDeleteConversation={vi.fn()}
        onNewChat={vi.fn()}
      />
    );

    const activeItem = screen.getByRole('button', { name: /multi-head attention query/i });
    expect(activeItem).toHaveAttribute('aria-current', 'true');

    const inactiveItem = screen.getByRole('button', {
      name: /segment-level recurrence in xl/i,
    });
    expect(inactiveItem).not.toHaveAttribute('aria-current', 'true');
  });

  it('invokes onSelectConversation when conversation item is clicked', () => {
    const handleSelect = vi.fn();
    render(
      <ConversationList
        conversations={mockConversations}
        activeConversationId={null}
        onSelectConversation={handleSelect}
        onDeleteConversation={vi.fn()}
        onNewChat={vi.fn()}
      />
    );

    const item = screen.getByRole('button', { name: /multi-head attention query/i });
    fireEvent.click(item);

    expect(handleSelect).toHaveBeenCalledTimes(1);
    expect(handleSelect).toHaveBeenCalledWith(mockConversations[0]);
  });

  it('invokes onDeleteConversation when delete button is clicked and stops propagation', () => {
    const handleSelect = vi.fn();
    const handleDelete = vi.fn();

    render(
      <ConversationList
        conversations={mockConversations}
        activeConversationId="conv_1"
        onSelectConversation={handleSelect}
        onDeleteConversation={handleDelete}
        onNewChat={vi.fn()}
      />
    );

    const deleteButtons = screen.getAllByRole('button', { name: /delete conversation/i });
    expect(deleteButtons).toHaveLength(2);

    fireEvent.click(deleteButtons[0]);

    expect(handleDelete).toHaveBeenCalledTimes(1);
    expect(handleDelete).toHaveBeenCalledWith('conv_1');
    expect(handleSelect).not.toHaveBeenCalled();
  });

  it('invokes onNewChat when "+ New Chat" button is clicked', () => {
    const handleNewChat = vi.fn();

    render(
      <ConversationList
        conversations={mockConversations}
        activeConversationId="conv_1"
        onSelectConversation={vi.fn()}
        onDeleteConversation={vi.fn()}
        onNewChat={handleNewChat}
      />
    );

    const newChatBtn = screen.getByRole('button', { name: /new chat/i });
    fireEvent.click(newChatBtn);

    expect(handleNewChat).toHaveBeenCalledTimes(1);
  });

  it('renders loading skeleton when isLoading is true', () => {
    render(
      <ConversationList
        conversations={[]}
        activeConversationId={null}
        isLoading={true}
        onSelectConversation={vi.fn()}
        onDeleteConversation={vi.fn()}
        onNewChat={vi.fn()}
      />
    );

    expect(screen.getByTestId('conversation-list-loading')).toBeInTheDocument();
  });

  it('toggles section collapse when accordion header is clicked', () => {
    render(
      <ConversationList
        conversations={mockConversations}
        activeConversationId="conv_1"
        onSelectConversation={vi.fn()}
        onDeleteConversation={vi.fn()}
        onNewChat={vi.fn()}
      />
    );

    expect(screen.getByText('Multi-Head Attention Query')).toBeInTheDocument();

    const headerToggle = screen.getByRole('button', { name: /toggle session history/i });
    fireEvent.click(headerToggle);

    // When collapsed, conversation items are hidden
    expect(screen.queryByText('Multi-Head Attention Query')).not.toBeInTheDocument();

    // Clicking again expands the section
    fireEvent.click(headerToggle);
    expect(screen.getByText('Multi-Head Attention Query')).toBeInTheDocument();
  });
});
