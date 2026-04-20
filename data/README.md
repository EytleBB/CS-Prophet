This repository now treats `data/` as a lightweight fallback and placeholder area.

Primary data storage lives under the external data root:

- `H:\CS_Prophet\data` on this machine
- or `CS_PROPHET_DATA_ROOT` if that environment variable is set

Code should resolve data paths through `src/utils/paths.py` instead of hardcoding
repo-local `data/...` paths.

Repo-local `data/` may still temporarily contain:

- small placeholder files such as `.gitkeep`
- lightweight assets that have not been migrated yet
- legacy outputs from before the external data-root migration
