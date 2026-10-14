import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { ChatHeader } from './ChatHeader';

describe('ChatHeader component', () => {
  it('renders active conversation title and system specs button', () => {
    render(
      <ChatHeader
        title="Attention Is All You Need"
        isSidebarOpen={true}
        onToggleSidebar={vi.fn()}
        onRename={vi.fn()}
        onOpenSpecs={vi.fn()}
      />
    );

    expect(screen.getByText('Attention Is All You Need')).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /system architecture & telemetry/i })
    ).toBeInTheDocument();
    expect(screen.getByText(/System Specs/i)).toBeInTheDocument();
  });

  it('renders expand sidebar button when sidebar is collapsed and triggers onToggleSidebar', () => {
    const handleToggle = vi.fn();
    render(
      <ChatHeader
        title="New Chat"
        isSidebarOpen={false}
        onToggleSidebar={handleToggle}
        onRename={vi.fn()}
        onOpenSpecs={vi.fn()}
      />
    );

    const expandBtn = screen.getByRole('button', { name: /expand documents tray/i });
    expect(expandBtn).toBeInTheDocument();

    fireEvent.click(expandBtn);
    expect(handleToggle).toHaveBeenCalledTimes(1);
  });

  it('does not render expand sidebar button when sidebar is open', () => {
    render(
      <ChatHeader
        title="New Chat"
        isSidebarOpen={true}
        onToggleSidebar={vi.fn()}
        onRename={vi.fn()}
        onOpenSpecs={vi.fn()}
      />
    );

    expect(
      screen.queryByRole('button', { name: /expand documents tray/i })
    ).not.toBeInTheDocument();
  });

  it('triggers onOpenSpecs when system specs trigger is clicked', () => {
    const handleOpenSpecs = vi.fn();
    render(
      <ChatHeader
        title="New Chat"
        isSidebarOpen={true}
        onToggleSidebar={vi.fn()}
        onRename={vi.fn()}
        onOpenSpecs={handleOpenSpecs}
      />
    );

    const specsBtn = screen.getByRole('button', { name: /system architecture & telemetry/i });
    fireEvent.click(specsBtn);
    expect(handleOpenSpecs).toHaveBeenCalledTimes(1);
  });

  it('switches to inline edit mode on click and saves on Enter', () => {
    const handleRename = vi.fn();
    render(
      <ChatHeader
        title="Old Session Title"
        isSidebarOpen={true}
        onToggleSidebar={vi.fn()}
        onRename={handleRename}
        onOpenSpecs={vi.fn()}
      />
    );

    // Click on rename trigger or title
    const titleBtn = screen.getByRole('button', { name: /rename conversation/i });
    fireEvent.click(titleBtn);

    const input = screen.getByRole('textbox', { name: /edit conversation title/i });
    expect(input).toBeInTheDocument();
    expect(input).toHaveValue('Old Session Title');

    fireEvent.change(input, { target: { value: 'New Paper Discussion' } });
    fireEvent.keyDown(input, { key: 'Enter' });

    expect(handleRename).toHaveBeenCalledWith('New Paper Discussion');
    expect(
      screen.queryByRole('textbox', { name: /edit conversation title/i })
    ).not.toBeInTheDocument();
  });

  it('cancels inline edit on Escape key without calling onRename', () => {
    const handleRename = vi.fn();
    render(
      <ChatHeader
        title="Original Title"
        isSidebarOpen={true}
        onToggleSidebar={vi.fn()}
        onRename={handleRename}
        onOpenSpecs={vi.fn()}
      />
    );

    const titleBtn = screen.getByRole('button', { name: /rename conversation/i });
    fireEvent.click(titleBtn);

    const input = screen.getByRole('textbox', { name: /edit conversation title/i });
    fireEvent.change(input, { target: { value: 'Changed But Canceled' } });
    fireEvent.keyDown(input, { key: 'Escape' });

    expect(handleRename).not.toHaveBeenCalled();
    expect(
      screen.queryByRole('textbox', { name: /edit conversation title/i })
    ).not.toBeInTheDocument();
    expect(screen.getByText('Original Title')).toBeInTheDocument();
  });

  it('saves on blur if title changed', () => {
    const handleRename = vi.fn();
    render(
      <ChatHeader
        title="Original Title"
        isSidebarOpen={true}
        onToggleSidebar={vi.fn()}
        onRename={handleRename}
        onOpenSpecs={vi.fn()}
      />
    );

    const titleBtn = screen.getByRole('button', { name: /rename conversation/i });
    fireEvent.click(titleBtn);

    const input = screen.getByRole('textbox', { name: /edit conversation title/i });
    fireEvent.change(input, { target: { value: 'Saved On Blur' } });
    fireEvent.blur(input);

    expect(handleRename).toHaveBeenCalledWith('Saved On Blur');
  });
});
