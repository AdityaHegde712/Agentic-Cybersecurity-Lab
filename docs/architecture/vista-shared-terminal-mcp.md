# Vista Shared Terminal MCP
## Implementation Design for Codex

**Purpose:** Build a small local MCP server that lets Codex operate the exact same visible terminal session the human is watching while that terminal is connected to TACC Vista.

**Primary environment:** Local machine + visible terminal + `tmux` + SSH to Vista.

**Non-goal:** This project does not automate SSH authentication, TACC MFA, Slurm itself, or remote code editing.

---

## 1. Desired Behavior

The user's normal workflow is:

1. Develop code locally with Codex.
2. Push changes to Git.
3. Open a visible terminal.
4. SSH into Vista.
5. `git pull` on Vista.
6. Use Vista login nodes only for lightweight orchestration.
7. Enter an appropriate interactive Slurm allocation when required, including GG/GH compute resources.
8. Execute and inspect workloads on the allocated compute node.

Codex should be able to perform steps 4–8 through the visible terminal without creating a separate hidden remote shell.

The human and Codex must share the **same terminal/PTY**.

The human must be able to:
- see every command Codex enters;
- see all output in real time;
- type into the session at any time;
- interrupt a running command with Ctrl-C;
- detach/reattach without destroying the session.

Codex must be able to:
- inspect recent terminal output;
- type a command;
- press special keys;
- wait for output or prompts;
- observe command results;
- continue operating after the shell moves from local → Vista login node → allocated compute node.

---

## 2. Architecture

Use a local `tmux` session as the shared terminal substrate.

```text
                         +----------------------+
                         |        Codex         |
                         +----------+-----------+
                                    |
                                    | MCP over stdio
                                    v
                         +----------------------+
                         | vista-terminal-mcp   |
                         +----------+-----------+
                                    |
                                    | tmux CLI
                                    v
+------------------+     +----------------------+
| Visible terminal |<--->| tmux session/pane    |
| human attached   |     | e.g. vista:0.0       |
+------------------+     +----------+-----------+
                                    |
                                    | shell / PTY
                                    v
                           ssh Vista login node
                                    |
                                    | idev / srun / shell
                                    v
                            GG/GH compute node
```

The MCP server is strictly local.

It should not run a service on Vista.

The tool should operate on one configured `tmux` pane, and that same pane should be attached to the user's visible terminal.

---

## 3. Why tmux

`tmux` already provides the required low-level primitives:

```bash
tmux capture-pane -p -t vista:0.0 -S -150
tmux send-keys -t vista:0.0 -l -- 'git pull'
tmux send-keys -t vista:0.0 Enter
tmux send-keys -t vista:0.0 C-c
```

Advantages over GUI automation:

- no dependence on window focus;
- no screen-coordinate automation;
- no OCR;
- no terminal screenshot interpretation;
- no clipboard dependency;
- no hidden SSH connection;
- persistent terminal state;
- human and Codex can use the same shell concurrently;
- continues to work through nested interactive programs such as SSH and `idev`.

---

## 4. Scope

### In scope

Implement:

- session discovery;
- optional session creation;
- terminal output capture;
- literal text input;
- Enter and special-key input;
- bounded waiting for strings or regular expressions;
- minimal session/pane metadata;
- safe error handling;
- Codex MCP configuration;
- concise usage documentation;
- unit tests for local logic;
- integration tests against a local tmux shell;
- a manual Vista validation checklist.

### Out of scope

Do not implement:

- SSH password storage;
- MFA storage or automation;
- TOTP generation;
- remote Vista daemons;
- remote MCP servers;
- arbitrary GUI automation;
- computer vision;
- automatic Slurm policy decisions;
- code editing on Vista;
- hidden `ssh host command` execution;
- arbitrary selection/control of unrelated tmux sessions;
- automatic recovery from destructive or ambiguous terminal states.

---

## 5. MCP Interface

Keep the public interface intentionally small.

### 5.1 `terminal.status`

Returns information about the configured shared pane.

Suggested output:

```json
{
  "session": "vista",
  "target": "vista:0.0",
  "exists": true,
  "pane_dead": false,
  "attached_clients": 1,
  "pane_current_command": "ssh"
}
```

Useful checks:

```bash
tmux has-session -t vista
tmux display-message -p -t vista:0.0 '#{pane_dead}'
tmux display-message -p -t vista:0.0 '#{pane_current_command}'
tmux list-clients
```

Do not treat `pane_current_command` as a reliable indicator of the remote host. During a remote session it will usually just be `ssh`.

---

### 5.2 `terminal.read`

Read recent terminal contents.

Suggested input:

```json
{
  "lines": 150
}
```

Suggested implementation:

```bash
tmux capture-pane -p -t vista:0.0 -S -150
```

Requirements:

- default around 100–200 lines;
- configurable maximum;
- strip no meaningful shell output;
- preserve line ordering;
- return plain text;
- never capture unlimited history by default.

The terminal itself is the source of truth. Codex should read before acting whenever the state may have changed.

---

### 5.3 `terminal.send`

Send literal text and optionally Enter.

Suggested input:

```json
{
  "text": "git pull --ff-only",
  "enter": true
}
```

Implementation:

1. send the string literally;
2. send Enter separately if requested.

Use an argument-vector subprocess API, not shell interpolation.

Example conceptual calls:

```text
["tmux", "send-keys", "-t", TARGET, "-l", "--", TEXT]
["tmux", "send-keys", "-t", TARGET, "Enter"]
```

Do **not** do this:

```text
shell=True
bash -c ...
sh -c ...
```

User-supplied terminal text must never be interpreted by the local shell.

---

### 5.4 `terminal.key`

Send a special key.

Suggested input:

```json
{
  "key": "C-c"
}
```

Initial allow-list:

- `Enter`
- `C-c`
- `C-d`
- `C-z`
- `Tab`
- `Escape`
- `Up`
- `Down`
- `Left`
- `Right`

Reject unknown values.

Do not expose arbitrary tmux commands.

---

### 5.5 `terminal.wait`

Poll captured terminal output until text or a regex appears.

Suggested input:

```json
{
  "pattern": "TACC Token Code:",
  "regex": false,
  "timeout_seconds": 120
}
```

Suggested behavior:

1. capture a bounded recent region;
2. search;
3. sleep for a configured short interval;
4. repeat until match or timeout;
5. return whether matched plus the final capture.

Defaults:

```text
poll_interval_ms = 500
default_timeout_seconds = 60
maximum_timeout_seconds = 300
```

This is useful for:

- SSH prompts;
- TACC MFA handoff;
- allocation messages;
- compute-node prompts;
- workload completion text;
- errors.

`terminal.wait` is not a generic process supervisor. It only observes the shared terminal.

---

### 5.6 `terminal.create_session`

Optional convenience endpoint.

Behavior:

- create the configured session only if it does not already exist;
- create it detached;
- return the target name and attach command;
- never automatically launch SSH.

Conceptual command:

```bash
tmux new-session -d -s vista
```

Human attachment:

```bash
tmux attach -t vista
```

---

## 6. Configuration

Use a small config file or environment variables.

Example:

```toml
session_name = "vista"
window_index = 0
pane_index = 0

default_read_lines = 150
max_read_lines = 2000

default_wait_seconds = 60
max_wait_seconds = 300
poll_interval_ms = 500
```

Derived target:

```text
vista:0.0
```

Do not include:

- TACC password;
- MFA token or seed;
- private SSH keys;
- Git credentials;
- secrets from shell startup files.

---

## 7. Concurrency Model

Both the user and Codex can type into the shared shell.

This must be treated as cooperative concurrent control.

### Codex operating rules

1. **Read before acting.**
   If meaningful time has passed, or the user may have interacted with the terminal, call `terminal.read`.

2. **Send one complete line atomically.**
   Avoid simulated human character-by-character typing.

3. **Inspect results before issuing dependent commands.**
   Do not send a chain of commands as independent blind steps.

4. **Human state wins.**
   If the terminal has changed unexpectedly, adapt to the latest observed state.

5. **Never assume exclusive ownership.**

6. **Do not infer the current machine solely from previous actions.**
   When it matters, explicitly run `hostname`.

7. **Do not infer the working directory solely from memory.**
   When it matters, explicitly run `pwd`.

---

## 8. Recommended Codex Instructions

Add project-level instructions similar to:

```text
When operating TACC Vista, use the shared Vista terminal MCP. Do not open a
separate SSH session unless the human explicitly asks you to.

Before acting, inspect the terminal if its current state is uncertain.

After sending a consequential command, inspect its output before continuing.

The human can type into the shared terminal at any time. Treat the latest
terminal contents as authoritative.

Do not run compute-heavy workloads on Vista login nodes. Use the appropriate
Slurm allocation or interactive compute-node workflow first.

If an SSH password or MFA prompt is visible, do not attempt to fill it.
Tell the human that authentication is required and wait for the terminal to
advance.

Do not intentionally capture credentials while the human is typing them.

Never issue destructive commands based only on inferred state. Verify host,
working directory, and relevant target first.
```

---

## 9. Authentication Handoff

Authentication itself is not part of this MCP.

Expected interaction:

```text
Codex -> terminal.send("ssh vista", enter=true)
Codex -> terminal.wait("TACC Token Code:")
Human -> enters MFA code in visible terminal
Codex -> terminal.wait(successful-login condition)
Codex -> terminal.read()
```

If a password prompt occurs, the same rule applies: the human enters it.

Codex should not call `terminal.read` repeatedly while the human is actively entering a secret.

---

## 10. Vista / HPC Operating Model

The MCP transport should not encode HPC policy.

Codex should reason from the shell state and user's instructions.

Typical state progression:

```text
LOCAL
  |
  +-- ssh vista
  v
VISTA LOGIN NODE
  |
  +-- lightweight git/status/Slurm operations
  |
  +-- idev / appropriate allocation
  v
COMPUTE NODE
  |
  +-- execute workload
  +-- inspect output
```

When the current state matters, Codex can explicitly query it:

```bash
hostname
pwd
git status --short --branch
squeue -u "$USER"
```

Avoid building a complex hard-coded state machine unless real usage demonstrates a need.

---

## 11. Safety Requirements

### Target pinning

By default the MCP server controls exactly one configured pane.

Do not expose arbitrary pane/session selection in v1.

### No local-shell interpolation

All `tmux` commands must use safe subprocess argument arrays.

Terminal input is data.

### Input validation

Validate:

- line counts;
- timeout values;
- regex compilation;
- special-key names;
- target configuration.

### Bounded resource use

Cap:

- capture history;
- wait duration;
- poll rate;
- returned output size.

### No secret handling

The MCP server should never request, store, or log:

- passwords;
- MFA tokens;
- TOTP seeds;
- private keys.

### Visible execution only

Do not add a hidden SSH execution path to this tool.

The purpose of this project is shared visible terminal operation.

### User takeover

The human must always retain the ability to type or send Ctrl-C.

Do not implement terminal locking.

---

## 12. Logging

Default logs should contain MCP operational metadata, not terminal content.

Example:

```text
terminal.read lines=150 success=true
terminal.send chars=18 enter=true success=true
terminal.wait regex=false timeout=60 matched=true
terminal.key key=C-c success=true
```

Do not log exact commands or captured terminal output by default.

Verbose debugging may optionally do so, but must be opt-in and clearly warn that terminal content may include sensitive data.

---

## 13. Error Model

Return useful structured errors.

Examples:

### tmux missing

```text
code: TMUX_NOT_FOUND
message: tmux is not installed or not available on PATH.
```

### session missing

```text
code: SESSION_NOT_FOUND
message: Configured tmux session 'vista' does not exist.
```

### pane missing

```text
code: PANE_NOT_FOUND
message: Configured target 'vista:0.0' does not exist.
```

### dead pane

```text
code: PANE_DEAD
message: Configured pane is no longer running.
```

### invalid key

```text
code: INVALID_KEY
message: Requested special key is not in the allowed set.
```

### invalid regex

```text
code: INVALID_PATTERN
message: Pattern could not be compiled as a regular expression.
```

### wait timeout

```text
code: WAIT_TIMEOUT
message: Pattern was not observed before the configured timeout.
```

A timeout must not be treated as proof that the remote command failed.

---

## 14. Internal Structure

Keep the implementation small.

Suggested modules:

```text
src/
  server
  config
  terminal/
    tmux_adapter
    wait
  errors
tests/
  unit/
  integration/
```

Conceptual class:

```text
TmuxTerminal
  session_exists()
  pane_exists()
  create_session()
  status()
  capture(lines)
  send_text(text, enter)
  send_key(key)
  wait_for(pattern, regex, timeout)
```

The MCP layer should mostly validate arguments and delegate to this adapter.

---

## 15. Testing Strategy

Fully automated end-to-end Vista testing is neither required nor desirable because authentication, MFA, Slurm allocation, queue timing, and compute-node assignment are external interactive state.

However, most of the MCP can and should be tested automatically **without Vista**.

### 15.1 Unit tests

Test:

- config parsing;
- target derivation;
- line-count bounds;
- timeout bounds;
- allowed special keys;
- invalid keys;
- regex compilation;
- wait matching logic;
- timeout logic;
- subprocess argument construction;
- error translation.

No real SSH connection is needed.

### 15.2 Local integration tests

Create a disposable local tmux session.

Test:

1. session creation;
2. pane discovery;
3. literal text send;
4. Enter;
5. output capture;
6. special keys;
7. wait-for-string;
8. wait-for-regex;
9. timeout;
10. unusual shell characters.

Important test strings should include shell metacharacters:

```text
$HOME
$(echo unsafe)
`echo unsafe`
"quotes"
'single quotes'
&&
;
|
>
<
*
?
```

The test should confirm they are inserted into the terminal as literal text and are **not interpreted by the local MCP process**.

Where executing them inside the test shell would be dangerous or ambiguous, insert them without Enter and inspect the terminal contents.

### 15.3 Concurrency sanity test

While Codex/test code is using the pane:

1. human manually types a command;
2. MCP captures it;
3. MCP sends a subsequent command;
4. verify both share the same shell and history.

### 15.4 Persistence test

1. attach to the pane;
2. detach;
3. interact through MCP;
4. reattach;
5. confirm the same shell/session/output persists.

---

## 16. Manual Vista Validation

The human should be present for these tests.

Do not automate MFA.

### Test A — connect

1. Start local shared session.
2. Attach visibly.
3. Have Codex send the normal Vista SSH command.
4. Human completes authentication.
5. Codex reads the successful login output.
6. Codex sends:

```bash
hostname
```

**Pass:** command visibly appears and Vista hostname is captured.

### Test B — Git operation

Codex:

```bash
cd <known-test-repository>
git status --short --branch
git pull --ff-only
```

**Pass:** commands and output are visible and correctly captured.

### Test C — interactive allocation

Have Codex enter the user's normal approved `idev`/interactive Slurm command.

Codex waits for allocation completion.

Then run:

```bash
hostname
```

**Pass:** terminal transitions to the allocated node and Codex correctly observes the new host.

### Test D — workload

Run a harmless existing script or trivial compute-node command.

**Pass:** output is visible to both human and Codex.

### Test E — human takeover

While a harmless long-running command executes, the human manually presses Ctrl-C.

**Pass:** Codex's next read sees the interruption and proceeds from the new state.

### Test F — Codex interrupt

Run another harmless wait/sleep command.

Have Codex call:

```text
terminal.key("C-c")
```

**Pass:** the visible command is interrupted immediately.

### Test G — detach/re-attach

1. detach visible tmux client;
2. keep the session alive;
3. have Codex interact with it;
4. reattach.

**Pass:** same remote shell remains active and the intervening commands/output are visible.

---

## 17. Acceptance Criteria

The implementation is complete when all of the following are true:

- [ ] Codex can discover/create the configured tmux session.
- [ ] The human can attach to the exact pane Codex controls.
- [ ] Codex can read recent terminal contents.
- [ ] Codex can send literal text safely.
- [ ] Codex can send Enter separately.
- [ ] Codex can send Ctrl-C and the supported special keys.
- [ ] Codex can wait for literal text.
- [ ] Codex can wait for regex patterns.
- [ ] Waits are bounded and cancel on timeout.
- [ ] Terminal capture is bounded.
- [ ] Arbitrary terminal text is never interpolated through a local shell.
- [ ] The server is pinned to the intended tmux target.
- [ ] Unit tests pass.
- [ ] Local tmux integration tests pass.
- [ ] Vista SSH manual test passes.
- [ ] Git pull manual test passes.
- [ ] Interactive compute-node transition test passes.
- [ ] Human takeover works.
- [ ] Codex Ctrl-C works.
- [ ] Detach/re-attach preserves the shared session.
- [ ] No SSH password or MFA automation exists.
- [ ] No hidden remote-execution path exists.

---

## 18. Suggested First Version

Do not over-engineer v1.

Implement only:

```text
terminal.status
terminal.read
terminal.send
terminal.key
terminal.wait
terminal.create_session
```

Use one configured pane.

Use synchronous subprocess execution for short `tmux` commands.

Use a simple polling loop for `terminal.wait`.

Add sophisticated prompt detection, command framing, multiple-pane support, or remote-state tracking only if real usage demonstrates that they are necessary.

---

## 19. Potential Follow-Ups

Possible later improvements, deliberately excluded from v1:

- multiple named terminal targets;
- automatic detection of whether the human is currently typing;
- command completion markers;
- terminal output deltas instead of repeated captures;
- optional pane creation/window management;
- structured Slurm helper tools;
- safe confirmation gates for destructive commands;
- persistent event stream from tmux;
- richer remote-host detection.

Do not implement these until the simple shared-terminal workflow has been used successfully on Vista.

---

## 20. Core Design Principle

The MCP server is not a remote execution system.

It is a **small adapter that gives Codex controlled keyboard-and-screen access to a terminal session already owned and visible to the human**.

`tmux` owns the PTY.

The human and Codex are peers operating the same session.

The terminal remains the authoritative record of state.
