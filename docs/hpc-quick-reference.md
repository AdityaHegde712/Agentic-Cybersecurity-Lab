# TACC Vista Personal Cheatsheet & Quick Reference

Quick operational cheatsheet for daily development, connecting, copying files, and launching compute sessions on TACC Vista.

---

## 1. Local Machine Quick Aliases (`~/.bashrc`)

### Connect to Vista
```bash
# Add to ~/.bashrc:
alias vistaconnect='ssh adityahegde712@vista.tacc.utexas.edu'

# Connect:
vistaconnect
```

### Copy Files to Vista (`vistacopy`)
```bash
# Add to ~/.bashrc:
vistacopy() {
    local source=""
    local destination=""
    local remote_base="adityahegde712@vista.tacc.utexas.edu:/work/11784/adityahegde712/vista/Agentic-Cybersecurity-Lab"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --source|-s) source="$2"; shift 2 ;;
            --destination|-d) destination="$2"; shift 2 ;;
            *)
                if [[ -z "$source" ]]; then source="$1"
                elif [[ -z "$destination" ]]; then destination="$1"; fi
                shift ;;
        esac
    done

    if [[ -z "$source" || -z "$destination" ]]; then
        echo "Usage: vistacopy --source <src_folder> --destination <relative_dest>"
        return 1
    fi
    destination="${destination#/}"
    scp -r "$source" "${remote_base}/${destination}"
}

# Example Usages from Local Repository:
vistacopy --source data/processed --destination data/
vistacopy --source configs/default.yaml --destination configs/
```

---

## 2. Directory Navigation on Vista

| Target Directory | Quick Command | Full Path | Quota / Purge |
|---|---|---|---|
| **Repository Root** | `cd $WORK/Agentic-Cybersecurity-Lab` | `/work/11784/adityahegde712/vista/Agentic-Cybersecurity-Lab` | 1 TB (Persistent) |
| **Virtual Environments** | `cd $SCRATCH/venvs` | `/scratch/11784/adityahegde712/vista/venvs` | Fast Flash (10-day purge) |
| **Home Directory** | `cd $HOME` | `/home1/11784/adityahegde712` | 23 GB (Backed up) |

---

## 3. Environment & Python Activation

> [!NOTE]
> Packages are permanently installed in `$SCRATCH/venvs/sengupta_cyber`. However, each new shell, login, or `idev` compute node session starts with a clean environment and requires activating the environment.

### Quick Manual Activation (Per Session)
```bash
module load gcc cuda/13.1 python3/3.11.8
source $SCRATCH/venvs/sengupta_cyber/bin/activate
```

### Auto-Alias Setup on Vista (Run Once in Vista Shell)
Add this alias to your remote `~/.bashrc` on Vista so you can activate with a single word (`loadenv`):
```bash
echo "alias loadenv='module load gcc cuda/13.1 python3/3.11.8 && source \$SCRATCH/venvs/sengupta_cyber/bin/activate'" >> ~/.bashrc
source ~/.bashrc

# Now simply run in any new idev or terminal session:
loadenv
```

---

## 4. Interactive Node Execution (`idev`)

Always use allocation `TG-NAIRR260371` and select option **`3`** (`CONT. without using reservation`):

```bash
# Request 1 Grace-Hopper GPU node for 60 mins (Cost: 1.0 SU)
idev -p gh-dev -N 1 -n 1 -m 60 -A TG-NAIRR260371

# Request 1 Grace-Grace 144-core CPU node for 120 mins (Cost: 0.66 SUs)
idev -p gg -N 1 -n 144 -m 120 -A TG-NAIRR260371
```

---

## 5. Slurm Monitoring & Job Control

```bash
# View active jobs
squeue -u $USER
showq -u

# Check allocation queue limits
qlimits

# Cancel a specific job
scancel <JOB_ID>
```
