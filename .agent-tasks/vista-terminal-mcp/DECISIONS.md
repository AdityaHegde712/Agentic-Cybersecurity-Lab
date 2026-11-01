# Vista Shared Terminal MCP Decisions

## Accepted

- Python is the implementation language.
- The implementation is isolated at `C:/Users/hifia/.codex/mcp_servers/vista-terminal-mcp/` and does not belong to the research package or its dependency set.
- Codex globally registers `vista-shared-terminal` to that project's own virtual-environment interpreter, with socket `vista-shared` and target session `vista` supplied as process environment values.
- MSYS2 SSH configuration provides `ssh vista` for `adityahegde712@vista.tacc.utexas.edu:22`; authentication remains human-only.
- The MCP server is local and stdio-only.
- MSYS2 MSYS owns the visible shell and tmux server; the configured tmux executable is `C:/msys64/usr/bin/tmux.exe`.
- The server controls only socket `vista-shared` and pane `vista:0.0`.
- Password and MFA entry are human-only. The server must not read or log terminal contents during an authentication handoff.
- `KeyboardInterrupt` and `Keyboard Interrupted` in newly observed output pause automation until an explicit resume call made after a new owner instruction.
- Every `Enter` path has a fixed five-second human review cooldown. The server first waits until typed text is visible, then compares pane snapshots after the cooldown; any change cancels Enter.
- `rm` candidate invocations are hard-coded as prohibited before terminal text is typed. This is a conservative lexical guard for standard `rm`, `sudo rm`, and path-qualified `rm` forms.

## Rejected for v1

- Direct remote Codex app-server workflow, generic SSH MCP servers, and generic spawned-PTY servers.
- Multiple terminal targets, remote daemons, terminal locking, hidden SSH execution, Slurm-specific tools, and tmux control mode.
