# Vista Shared Terminal MCP Tasks

- [x] 1. Confirm MSYS2 tmux, Git, and SSH are available from one shell.
- [x] 2. Confirm a visible local `vista-shared` tmux session survives detach/reattach.
- [x] 3. Confirm a native Windows process can capture `vista:0.0` from the MSYS2 tmux socket.
- [x] 4. Write immutable specification tests for the pinned, bounded, interrupt-aware terminal contract.
  **Evidence:** `C:/Users/hifia/.codex/mcp_servers/vista-terminal-mcp/tests/spec/test_vista_terminal_contract.py` passed with all local tests; no assertion was weakened or removed.
- [x] 5. Implement the adapter, service, and stdio MCP tool layer.
  **Evidence:** package is isolated at `C:/Users/hifia/.codex/mcp_servers/vista-terminal-mcp/src/vista_terminal_mcp/`; local implementation never connected to Vista.
- [x] 6. Run a disposable local tmux integration suite and MCP stdio smoke.
  **Evidence:** 13 frozen contracts, 2 native-tmux integrations, and 1 MCP-stdio integration passed in `C:/Users/hifia/.codex/mcp_servers/vista-terminal-mcp/`. Native tmux and stdio tests used UUID-named sockets only and removed them during cleanup. A live-local `terminal.status` check confirmed `vista:0.0` is a live Bash pane with no pause latch.
- [/] 8. Add reviewed execution safety controls before any remote validation.
  **Acceptance:** each Enter has a fixed five-second review window that cancels on pane changes; candidate `rm` invocations are rejected before typing.
  **Evidence:** frozen guard contracts passed; native tmux and MCP stdio exercises passed separately.
- [ ] 7. Obtain owner approval for a manual Vista connection validation.
  **Status:** approved `ssh vista` was typed into `vista:0.0`, held for the five-second review window, and entered without a pane change. Local metadata reports the active command as `ssh`; authentication is awaiting human completion.
  **Acceptance:** human completes password/MFA; no credential text enters MCP output or logs.

## Immediate Next Action

Wait for owner confirmation that manual Vista authentication has completed; do not read pane output during authentication.
