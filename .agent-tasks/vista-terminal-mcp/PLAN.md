# Vista Shared Terminal MCP Plan

## Active Phase

Implement and locally validate a Python stdio MCP server that controls one pinned MSYS2 tmux pane. No Vista command is in scope until the local gates pass and the owner explicitly approves a manual authentication test.

## Architecture

`MCPServer tools -> TerminalService -> TmuxAdapter -> C:/msys64/usr/bin/tmux.exe -L vista-shared -> vista:0.0`

The adapter uses subprocess argument arrays and `asyncio.to_thread`. Terminal text is sent literally, special keys are allow-listed, capture and waits are bounded, and `KeyboardInterrupt` / `Keyboard Interrupted` latch automation until explicit resume.

## Affected Paths

- `C:/Users/hifia/.codex/mcp_servers/vista-terminal-mcp/`: self-contained global MCP project, package, lock, and virtual environment.
- `C:/Users/hifia/.codex/mcp_servers/vista-terminal-mcp/tests/spec/`: immutable behavioral contracts.
- `C:/Users/hifia/.codex/mcp_servers/vista-terminal-mcp/tests/integration/`: disposable local tmux checks.
- `VISTA_SHARED_TERMINAL_MCP_DESIGN.md`: implementation-aligned decision addendum after tests pass.

## Exit Criteria

1. Frozen unit contracts pass. **Met: 8 local tests passed from the global location, including MSYS-native status-format expansion.**
2. A disposable local tmux integration test passes through the native Windows tmux executable. **Met.**
3. The MCP server responds over stdio to status/read/send/key/wait/create-session calls. **Met.**
4. The user approves the exact Vista SSH command before manual authentication; no allocation command is sent without a later, separate approval.
