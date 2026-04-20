"""Direct CS2 memory reader for realtime inference."""

from __future__ import annotations

import json
import logging
import struct
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import pymem
import pymem.process

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
MEMORY_DATA_DIR = REPO_ROOT / "src" / "memory_reader"
OFFSETS_PATH = MEMORY_DATA_DIR / "offsets.json"
CLIENT_DLL_JSON = MEMORY_DATA_DIR / "client_dll.json"

ENTITY_LIST_STRIDE = 0x70
MAX_PLAYERS = 64
TEAM_NAMES = {2: "T", 3: "CT", 1: "SPEC", 0: "NONE"}
_INVALID_HANDLE = 0xFFFFFFFF
_TEAM_SCAN_LIMIT = 256
_CT_TEAM_NAMES = {"CT", "COUNTERTERRORIST", "COUNTER-TERRORIST", "COUNTER_TERRORIST"}
_T_TEAM_NAMES = {"T", "TERRORIST", "TERRORISTS"}
_SMOKE_DURATION_SECONDS = 18.0
_MOLOTOV_DURATION_SECONDS = 7.0
_MAX_INVENTORY_WEAPONS = 64
_MAX_INFERNO_POINTS = 64

# Derive CT/T match score from C_CSGameRules.m_iMatchStats_RoundResults — a
# 30-slot int32 array where each entry is a Source 2 RoundEndReason_t enum.
# Demo playback exposes no C_CSTeam entity in the entity list, so the old
# designer-name scan always returned 0,0. Using the gamerules array lets us
# recover cumulative scores in both live and demo modes.
_ROUND_RESULT_SLOTS = 30
_ROUND_RESULT_CT_WIN = {5, 6, 7, 8, 11, 12, 14, 17}
_ROUND_RESULT_T_WIN = {1, 3, 4, 9, 13, 15, 18}


def _require_offset(table: dict[str, Any], module_name: str, offset_name: str) -> int:
    try:
        return int(table[module_name][offset_name])
    except KeyError as exc:  # pragma: no cover - configuration failure
        raise RuntimeError(f"offsets.json missing {module_name}.{offset_name}") from exc


def _require_field(classes: dict[str, Any], class_name: str, field_name: str) -> int:
    try:
        return int(classes[class_name]["fields"][field_name])
    except KeyError as exc:  # pragma: no cover - configuration failure
        raise RuntimeError(f"client_dll.json missing {class_name}.{field_name}") from exc


def _optional_field(classes: dict[str, Any], class_name: str, field_name: str) -> int | None:
    field = classes.get(class_name, {}).get("fields", {}).get(field_name)
    return int(field) if field is not None else None


@lru_cache(maxsize=1)
def load_offsets() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load static module offsets and client class field offsets once."""
    with OFFSETS_PATH.open("r", encoding="utf-8") as handle:
        offsets = json.load(handle)
    with CLIENT_DLL_JSON.open("r", encoding="utf-8") as handle:
        classes = json.load(handle)["client.dll"]["classes"]
    return offsets, classes


def _safe_read_string(pm: pymem.Pymem, ptr: int, max_len: int = 64) -> str:
    if ptr == 0:
        return ""
    try:
        return pm.read_string(ptr, max_len)
    except Exception:
        return ""


def _clean_inline_string(text: str) -> str:
    return text.split("\x00", 1)[0].strip()


def read_vec3(pm: pymem.Pymem, addr: int) -> tuple[float, float, float]:
    """Read a Source2 Vector at the given address."""
    buf = pm.read_bytes(addr, 12)
    return struct.unpack("<fff", buf)


def read_player_name(pm: pymem.Pymem, ctrl: int, name_offset: int) -> str:
    """Probe string_t / CUtlString-like storage layouts for player names."""
    field_addr = ctrl + name_offset

    try:
        p1 = pm.read_longlong(field_addr)
        if 0x10000 < p1 < 0x7FFFFFFFFFFF:
            text = _clean_inline_string(pm.read_string(p1, 64))
            if text and text.isprintable():
                return text
    except Exception:
        pass

    try:
        p1 = pm.read_longlong(field_addr)
        if 0x10000 < p1 < 0x7FFFFFFFFFFF:
            p2 = pm.read_longlong(p1)
            if 0x10000 < p2 < 0x7FFFFFFFFFFF:
                text = _clean_inline_string(pm.read_string(p2, 64))
                if text and text.isprintable():
                    return text
    except Exception:
        pass

    try:
        text = _clean_inline_string(pm.read_string(field_addr, 64))
        if text and text.isprintable():
            return text
    except Exception:
        pass

    return ""


def resolve_handle(pm: pymem.Pymem, ent_list_base: int, handle: int) -> int:
    """Resolve a CHandle<T> through the verified Source2 entity-list layout."""
    if handle in (0, _INVALID_HANDLE):
        return 0

    index = handle & 0x7FFF
    block = pm.read_longlong(ent_list_base + 0x8 * (index >> 9) + 0x10)
    if block == 0:
        return 0

    return pm.read_longlong(block + ENTITY_LIST_STRIDE * (index & 0x1FF))


def entity_class_name(
    pm: pymem.Pymem,
    entity_ptr: int,
    classes: dict[str, Any] | None = None,
) -> str:
    """Read CEntityIdentity.m_designerName for an entity pointer."""
    if entity_ptr == 0:
        return ""
    if classes is None:
        _, classes = load_offsets()

    try:
        identity_offset = _require_field(classes, "CEntityInstance", "m_pEntity")
        designer_offset = _require_field(classes, "CEntityIdentity", "m_designerName")
        identity = pm.read_longlong(entity_ptr + identity_offset)
        if identity == 0:
            return ""
        name_ptr = pm.read_longlong(identity + designer_offset)
        if name_ptr == 0:
            return ""
        return _clean_inline_string(pm.read_string(name_ptr, 64))
    except Exception:
        return ""


class CS2MemoryReader:
    """Attach to CS2 and read player/map state directly from memory."""

    def __init__(
        self,
        pm: pymem.Pymem,
        client_base: int,
        engine_base: int,
        offsets: dict[str, Any],
        classes: dict[str, Any],
    ) -> None:
        self.pm = pm
        self.client_base = int(client_base)
        self.engine_base = int(engine_base)
        self.offsets = offsets
        self.classes = classes
        self._warned: set[str] = set()
        self._team_handles: dict[str, int] = {}
        self._smoke_seen_at: dict[int, tuple[int, float]] = {}
        self._inferno_seen_at: dict[int, tuple[int, float]] = {}
        self._synthetic_molotovs: list[dict[str, float | int]] = []
        self._molotov_projectile_state: dict[int, dict[str, float | int | bool]] = {}

        self._client_offsets = {
            "dwEntityList": _require_offset(offsets, "client.dll", "dwEntityList"),
            "dwGameRules": _require_offset(offsets, "client.dll", "dwGameRules"),
            "dwPlantedC4": _require_offset(offsets, "client.dll", "dwPlantedC4"),
            "dwWeaponC4": _require_offset(offsets, "client.dll", "dwWeaponC4"),
        }
        self._engine_offsets = {
            "dwBuildNumber": _require_offset(offsets, "engine2.dll", "dwBuildNumber"),
        }

        self._controller_fields = {
            name: _require_field(classes, "CCSPlayerController", name)
            for name in (
                "m_hPlayerPawn",
                "m_bPawnIsAlive",
                "m_iPawnHealth",
                "m_iPawnArmor",
                "m_bPawnHasHelmet",
                "m_bPawnHasDefuser",
                "m_pInGameMoneyServices",
            )
        }
        self._base_controller_fields = {
            name: _require_field(classes, "CBasePlayerController", name)
            for name in ("m_iszPlayerName", "m_steamID")
        }
        self._pawn_fields = {
            name: _require_field(classes, "C_CSPlayerPawn", name)
            for name in ("m_angEyeAngles", "m_bIsScoped", "m_bIsDefusing", "m_pClippingWeapon")
        }
        self._base_pawn_fields = {
            "m_pWeaponServices": _require_field(classes, "C_BasePlayerPawn", "m_pWeaponServices"),
        }
        self._base_entity_fields = {
            name: _require_field(classes, "C_BaseEntity", name)
            for name in ("m_iTeamNum", "m_pGameSceneNode", "m_hOwnerEntity")
        }
        self._scene_node_fields = {
            "m_vecAbsOrigin": _require_field(classes, "CGameSceneNode", "m_vecAbsOrigin"),
        }
        self._weapon_service_fields = {
            name: _require_field(classes, "CPlayer_WeaponServices", name)
            for name in ("m_hMyWeapons", "m_hActiveWeapon")
        }
        self._planted_c4_fields = {
            name: _require_field(classes, "C_PlantedC4", name)
            for name in ("m_bBombTicking", "m_bBombDefused", "m_nBombSite")
        }
        self._smoke_fields = {
            name: _require_field(classes, "C_SmokeGrenadeProjectile", name)
            for name in ("m_bDidSmokeEffect", "m_nSmokeEffectTickBegin", "m_vSmokeDetonationPos")
        }
        self._inferno_fields = {
            name: _require_field(classes, "C_Inferno", name)
            for name in ("m_bFireIsBurning", "m_nFireLifetime", "m_nFireEffectTickBegin", "m_firePositions", "m_fireCount")
        }
        self._grenade_fields = {
            name: _require_field(classes, "C_BaseGrenade", name)
            for name in ("m_bIsLive", "m_flDetonateTime", "m_hThrower")
        }
        self._grenade_projectile_fields = {
            name: _require_field(classes, "C_BaseCSGrenadeProjectile", name)
            for name in (
                "m_bExplodeEffectBegan",
                "m_nExplodeEffectTickBegin",
                "m_vecExplodeEffectOrigin",
                "vecLastTrailLinePos",
                "m_nBounces",
            )
        }
        self._molotov_projectile_fields = {
            "m_bIsIncGrenade": _require_field(classes, "C_MolotovProjectile", "m_bIsIncGrenade"),
        }
        self._money_fields = {
            "m_iAccount": _require_field(classes, "CCSPlayerController_InGameMoneyServices", "m_iAccount"),
        }
        self._game_rules_fields = classes.get("C_CSGameRules", {}).get("fields", {})
        self._game_rules_proxy_fields = classes.get("C_CSGameRulesProxy", {}).get("fields", {})
        self._team_fields = classes.get("C_Team", {}).get("fields", {})

    @classmethod
    def attach(cls) -> CS2MemoryReader:
        """Attach to a live cs2.exe process and load the offset JSON payloads."""
        try:
            pm = pymem.Pymem("cs2.exe")
        except pymem.exception.ProcessNotFound as exc:
            raise RuntimeError("cs2.exe not running") from exc
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(f"failed to attach to cs2.exe: {exc}") from exc

        client = pymem.process.module_from_name(pm.process_handle, "client.dll")
        if client is None:
            raise RuntimeError("client.dll not located in cs2.exe")

        engine = pymem.process.module_from_name(pm.process_handle, "engine2.dll")
        if engine is None:
            raise RuntimeError("engine2.dll not located in cs2.exe")

        offsets, classes = load_offsets()
        return cls(
            pm=pm,
            client_base=client.lpBaseOfDll,
            engine_base=engine.lpBaseOfDll,
            offsets=offsets,
            classes=classes,
        )

    def _warn_once(self, key: str, message: str, *args: Any) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        logger.warning(message, *args)

    def _read_optional_int(
        self,
        addr: int,
        class_name: str,
        field_name: str,
        *,
        warn_key: str | None = None,
        default: int | None = None,
    ) -> int | None:
        field_offset = _optional_field(self.classes, class_name, field_name)
        if field_offset is None:
            if warn_key is not None:
                self._warn_once(warn_key, "client_dll.json missing %s.%s", class_name, field_name)
            return default
        try:
            return int(self.pm.read_int(addr + field_offset))
        except Exception as exc:
            if warn_key is not None:
                self._warn_once(warn_key, "failed reading %s.%s: %s", class_name, field_name, exc)
            return default

    def _read_optional_float(
        self,
        addr: int,
        class_name: str,
        field_name: str,
        *,
        warn_key: str | None = None,
        default: float | None = None,
    ) -> float | None:
        field_offset = _optional_field(self.classes, class_name, field_name)
        if field_offset is None:
            if warn_key is not None:
                self._warn_once(warn_key, "client_dll.json missing %s.%s", class_name, field_name)
            return default
        try:
            return float(self.pm.read_float(addr + field_offset))
        except Exception as exc:
            if warn_key is not None:
                self._warn_once(warn_key, "failed reading %s.%s: %s", class_name, field_name, exc)
            return default

    def _read_optional_bool(
        self,
        addr: int,
        class_name: str,
        field_name: str,
        *,
        warn_key: str | None = None,
        default: bool = False,
    ) -> bool:
        field_offset = _optional_field(self.classes, class_name, field_name)
        if field_offset is None:
            if warn_key is not None:
                self._warn_once(warn_key, "client_dll.json missing %s.%s", class_name, field_name)
            return default
        try:
            return bool(self.pm.read_uchar(addr + field_offset))
        except Exception as exc:
            if warn_key is not None:
                self._warn_once(warn_key, "failed reading %s.%s: %s", class_name, field_name, exc)
            return default

    def _resolve_game_rules_ptr(self) -> int:
        try:
            ptr = self.pm.read_longlong(self.client_base + self._client_offsets["dwGameRules"])
        except Exception as exc:
            self._warn_once("dwGameRules.read", "failed reading client.dll.dwGameRules: %s", exc)
            return 0
        if ptr == 0:
            self._warn_once("dwGameRules.null", "client.dll.dwGameRules resolved to null")
            return 0

        candidates = [ptr]
        proxy_offset = self._game_rules_proxy_fields.get("m_pGameRules")
        if proxy_offset is not None:
            try:
                proxy_target = self.pm.read_longlong(ptr + int(proxy_offset))
            except Exception:
                proxy_target = 0
            if proxy_target:
                candidates.append(proxy_target)

        for candidate in candidates:
            if self._looks_like_game_rules(candidate):
                return candidate
        return candidates[0]

    def _looks_like_game_rules(self, addr: int) -> bool:
        total_rounds_offset = _optional_field(self.classes, "C_CSGameRules", "m_totalRoundsPlayed")
        round_time_offset = _optional_field(self.classes, "C_CSGameRules", "m_iRoundTime")
        if total_rounds_offset is None or round_time_offset is None:
            return addr != 0
        try:
            total_rounds = self.pm.read_int(addr + total_rounds_offset)
            round_time = self.pm.read_int(addr + round_time_offset)
        except Exception:
            return False
        return -1 <= total_rounds <= 60 and 0 <= round_time <= 180

    def _read_match_scores(self, rules_ptr: int) -> tuple[int, int]:
        """Derive CT/T scores from C_CSGameRules.m_iMatchStats_RoundResults.

        Demo playback exposes no C_CSTeam entity, so cumulative scores must
        be reconstructed from the per-round result enum array bounded by
        m_totalRoundsPlayed. Unknown enum values (e.g. GameStart=16, Draw=10)
        count toward neither side.
        """
        array_offset = _optional_field(self.classes, "C_CSGameRules", "m_iMatchStats_RoundResults")
        total_offset = _optional_field(self.classes, "C_CSGameRules", "m_totalRoundsPlayed")
        if array_offset is None or total_offset is None:
            self._warn_once(
                "match_scores.fields",
                "client_dll.json missing m_iMatchStats_RoundResults or m_totalRoundsPlayed",
            )
            return 0, 0

        try:
            total_rounds = int(self.pm.read_int(rules_ptr + total_offset))
        except Exception:
            total_rounds = 0
        total_rounds = max(0, min(total_rounds, _ROUND_RESULT_SLOTS))
        if total_rounds == 0:
            return 0, 0

        try:
            raw = self.pm.read_bytes(rules_ptr + array_offset, total_rounds * 4)
        except Exception as exc:
            self._warn_once(
                "match_scores.read",
                "failed reading RoundResults array: %s",
                exc,
            )
            return 0, 0

        values = struct.unpack(f"<{total_rounds}i", raw)
        ct = sum(1 for v in values if v in _ROUND_RESULT_CT_WIN)
        t = sum(1 for v in values if v in _ROUND_RESULT_T_WIN)
        return ct, t

    def _read_team_scores(self, ent_list_base: int) -> tuple[int, int]:
        score_offset = _optional_field(self.classes, "C_Team", "m_iScore")
        name_offset = _optional_field(self.classes, "C_Team", "m_szTeamname")
        if score_offset is None or name_offset is None:
            self._warn_once(
                "team_scores.fields",
                "client_dll.json missing C_Team score/name fields; team scores will default to 0",
            )
            return 0, 0

        scores: dict[str, int] = {}
        found_handles: dict[str, int] = {}

        for side, team_handle in tuple(self._team_handles.items()):
            entity_ptr = resolve_handle(self.pm, ent_list_base, team_handle)
            if entity_ptr == 0:
                continue
            team_name = _clean_inline_string(
                _safe_read_string(self.pm, entity_ptr + name_offset, 32)
            ).upper()
            if not team_name:
                continue
            try:
                score_value = max(0, int(self.pm.read_int(entity_ptr + score_offset)))
            except Exception:
                continue
            scores[side] = score_value
            found_handles[side] = team_handle

        if "CT" in scores and "T" in scores:
            self._team_handles = found_handles
            return scores["CT"], scores["T"]

        for handle in range(1, _TEAM_SCAN_LIMIT + 1):
            entity_ptr = resolve_handle(self.pm, ent_list_base, handle)
            if entity_ptr == 0:
                continue

            team_name = _clean_inline_string(
                _safe_read_string(self.pm, entity_ptr + name_offset, 32)
            ).upper()
            if team_name in _CT_TEAM_NAMES:
                side = "CT"
            elif team_name in _T_TEAM_NAMES:
                side = "T"
            else:
                continue

            try:
                score_value = max(0, int(self.pm.read_int(entity_ptr + score_offset)))
            except Exception:
                continue

            scores[side] = score_value
            found_handles[side] = handle
            if "CT" in scores and "T" in scores:
                self._team_handles = found_handles
                return scores["CT"], scores["T"]

        self._warn_once(
            "team_scores.scan",
            "unable to resolve live C_Team scores from the entity list; team scores will default to 0",
        )
        self._team_handles = found_handles
        return scores.get("CT", 0), scores.get("T", 0)

    def _read_map_name(self, rules_ptr: int) -> str:
        for field_name in ("m_nMapName", "m_szMapName", "m_mapName"):
            field_offset = _optional_field(self.classes, "C_CSGameRules", field_name)
            if field_offset is None:
                continue
            text = _clean_inline_string(_safe_read_string(self.pm, rules_ptr + field_offset, 64))
            if text:
                return text

        self._warn_once(
            "map_name.field",
            "no C_CSGameRules map-name field found in client_dll.json; map_name will default to empty",
        )
        return ""

    def _read_entity_list_base(self) -> int:
        try:
            return int(self.pm.read_longlong(self.client_base + self._client_offsets["dwEntityList"]))
        except Exception as exc:
            self._warn_once("dwEntityList.read", "failed reading client.dll.dwEntityList: %s", exc)
            return 0

    def _read_utl_vector(self, addr: int) -> tuple[int, int]:
        try:
            size = int(self.pm.read_int(addr))
            elems = int(self.pm.read_longlong(addr + 8))
        except Exception:
            return 0, 0
        if size <= 0 or elems == 0:
            return 0, 0
        return size, elems

    def _read_entity_origin(self, entity_ptr: int) -> tuple[float, float, float]:
        if entity_ptr == 0:
            return 0.0, 0.0, 0.0
        try:
            scene_node = self.pm.read_longlong(entity_ptr + self._base_entity_fields["m_pGameSceneNode"])
            if scene_node:
                return read_vec3(self.pm, scene_node + self._scene_node_fields["m_vecAbsOrigin"])
        except Exception:
            pass
        return 0.0, 0.0, 0.0

    @staticmethod
    def _empty_inventory() -> dict[str, Any]:
        return {
            "weapons": [],
            "active_weapon_class": None,
            "has_smoke": False,
            "has_flash": False,
            "has_he": False,
            "has_molotov": False,
            "has_c4": False,
        }

    def _read_player_inventory_with_entity_list(
        self,
        pawn: int,
        ent_list_base: int,
    ) -> dict[str, Any]:
        inventory = self._empty_inventory()
        if pawn == 0 or ent_list_base == 0:
            return inventory

        try:
            weapon_services = self.pm.read_longlong(pawn + self._base_pawn_fields["m_pWeaponServices"])
        except Exception:
            return inventory
        if weapon_services == 0:
            return inventory

        size, elems = self._read_utl_vector(weapon_services + self._weapon_service_fields["m_hMyWeapons"])
        weapons: list[str] = []
        for slot in range(min(size, _MAX_INVENTORY_WEAPONS)):
            try:
                handle = int(self.pm.read_uint(elems + 4 * slot))
            except Exception:
                continue
            weapon_ptr = resolve_handle(self.pm, ent_list_base, handle)
            if weapon_ptr == 0:
                continue
            class_name = entity_class_name(self.pm, weapon_ptr, self.classes)
            if class_name:
                weapons.append(class_name)

        active_weapon_ptr = 0
        try:
            active_weapon_ptr = int(self.pm.read_longlong(pawn + self._pawn_fields["m_pClippingWeapon"]))
        except Exception:
            active_weapon_ptr = 0
        if active_weapon_ptr == 0:
            try:
                active_handle = int(
                    self.pm.read_uint(weapon_services + self._weapon_service_fields["m_hActiveWeapon"])
                )
            except Exception:
                active_handle = 0
            active_weapon_ptr = resolve_handle(self.pm, ent_list_base, active_handle)

        active_weapon_class = None
        if active_weapon_ptr:
            class_name = entity_class_name(self.pm, active_weapon_ptr, self.classes)
            if class_name:
                active_weapon_class = class_name

        weapon_set = set(weapons)
        inventory["weapons"] = weapons
        inventory["active_weapon_class"] = active_weapon_class
        inventory["has_smoke"] = "weapon_smokegrenade" in weapon_set
        inventory["has_flash"] = "weapon_flashbang" in weapon_set
        inventory["has_he"] = "weapon_hegrenade" in weapon_set
        inventory["has_molotov"] = (
            "weapon_molotov" in weapon_set or "weapon_incgrenade" in weapon_set
        )
        inventory["has_c4"] = "weapon_c4" in weapon_set
        return inventory

    def _projectile_remain(
        self,
        cache: dict[int, tuple[int, float]],
        entity_ptr: int,
        tick_begin: int,
        duration_seconds: float,
    ) -> float:
        if duration_seconds <= 0.0:
            return 0.0

        now = time.monotonic()
        cached = cache.get(entity_ptr)
        if cached is None or cached[0] != tick_begin:
            cache[entity_ptr] = (tick_begin, now)
            return 1.0

        elapsed = max(0.0, now - cached[1])
        return max(0.0, 1.0 - elapsed / duration_seconds)

    @staticmethod
    def _trim_projectile_cache(
        cache: dict[int, tuple[int, float]],
        active_entities: set[int],
    ) -> None:
        stale = [entity_ptr for entity_ptr in cache if entity_ptr not in active_entities]
        for entity_ptr in stale:
            cache.pop(entity_ptr, None)

    def _seed_synthetic_molotov(
        self,
        *,
        x: float,
        y: float,
        duration_seconds: float,
        signature: int,
    ) -> None:
        if x == 0.0 and y == 0.0:
            return

        now = time.monotonic()
        duration = max(1.0, float(duration_seconds or _MOLOTOV_DURATION_SECONDS))
        for item in self._synthetic_molotovs:
            item_x = float(item.get("x", 0.0))
            item_y = float(item.get("y", 0.0))
            item_signature = int(item.get("signature", 0))
            item_started = float(item.get("started_at", 0.0))
            item_duration = max(1.0, float(item.get("duration", _MOLOTOV_DURATION_SECONDS)))
            if item_signature and signature and item_signature == signature:
                return
            if abs(item_x - x) <= 96.0 and abs(item_y - y) <= 96.0 and (now - item_started) <= item_duration:
                return

        self._synthetic_molotovs.append(
            {
                "x": float(x),
                "y": float(y),
                "started_at": now,
                "duration": duration,
                "signature": int(signature),
            }
        )

    def _active_synthetic_molotovs(self) -> list[tuple[float, float, float]]:
        now = time.monotonic()
        active: list[tuple[float, float, float]] = []
        keep: list[dict[str, float | int]] = []
        for item in self._synthetic_molotovs:
            duration = max(1.0, float(item.get("duration", _MOLOTOV_DURATION_SECONDS)))
            started_at = float(item.get("started_at", 0.0))
            elapsed = max(0.0, now - started_at)
            remain = max(0.0, 1.0 - elapsed / duration)
            if remain <= 0.0:
                continue
            keep.append(item)
            active.append((float(item.get("x", 0.0)), float(item.get("y", 0.0)), remain))
        self._synthetic_molotovs = keep
        active.sort(key=lambda entry: (-entry[2], entry[0], entry[1]))
        return active

    def _read_inferno_candidate(self, entity_ptr: int, class_name: str) -> dict[str, Any] | None:
        lowered = class_name.lower()
        if not any(token in lowered for token in ("inferno", "molotov", "incendiary")):
            return None

        inferno_parent_positions = _optional_field(self.classes, "C_Inferno", "m_fireParentPositions")
        inferno_post_effect = _optional_field(self.classes, "C_Inferno", "m_bInPostEffectTime")
        inferno_type_field = _optional_field(self.classes, "C_Inferno", "m_nInfernoType")
        inferno_min_bounds = _optional_field(self.classes, "C_Inferno", "m_minBounds")
        inferno_max_bounds = _optional_field(self.classes, "C_Inferno", "m_maxBounds")

        is_burning = False
        in_post_effect = False
        fire_count = 0
        tick_begin = 0
        fire_lifetime = 0.0
        inferno_type = -1
        explode_effect_began = False
        is_live = False
        is_inc_grenade = False
        bounces = 0
        synthetic_seedable = False

        if "inferno" in lowered:
            try:
                is_burning = bool(
                    self.pm.read_uchar(entity_ptr + self._inferno_fields["m_bFireIsBurning"])
                )
            except Exception:
                is_burning = False
            if inferno_post_effect is not None:
                try:
                    in_post_effect = bool(self.pm.read_uchar(entity_ptr + inferno_post_effect))
                except Exception:
                    in_post_effect = False
            try:
                fire_count = max(
                    0,
                    min(
                        int(self.pm.read_int(entity_ptr + self._inferno_fields["m_fireCount"])),
                        _MAX_INFERNO_POINTS,
                    ),
                )
            except Exception:
                fire_count = 0
            try:
                tick_begin = int(self.pm.read_int(entity_ptr + self._inferno_fields["m_nFireEffectTickBegin"]))
            except Exception:
                tick_begin = 0
            try:
                fire_lifetime = float(self.pm.read_float(entity_ptr + self._inferno_fields["m_nFireLifetime"]))
            except Exception:
                fire_lifetime = 0.0
            if inferno_type_field is not None:
                try:
                    inferno_type = int(self.pm.read_int(entity_ptr + inferno_type_field))
                except Exception:
                    inferno_type = -1
        elif lowered == "molotov_projectile":
            try:
                explode_effect_began = bool(
                    self.pm.read_uchar(entity_ptr + self._grenade_projectile_fields["m_bExplodeEffectBegan"])
                )
            except Exception:
                explode_effect_began = False
            try:
                tick_begin = int(
                    self.pm.read_int(entity_ptr + self._grenade_projectile_fields["m_nExplodeEffectTickBegin"])
                )
            except Exception:
                tick_begin = 0
            try:
                is_live = bool(self.pm.read_uchar(entity_ptr + self._grenade_fields["m_bIsLive"]))
            except Exception:
                is_live = False
            try:
                is_inc_grenade = bool(
                    self.pm.read_uchar(entity_ptr + self._molotov_projectile_fields["m_bIsIncGrenade"])
                )
            except Exception:
                is_inc_grenade = False
            try:
                bounces = max(0, int(self.pm.read_int(entity_ptr + self._grenade_projectile_fields["m_nBounces"])))
            except Exception:
                bounces = 0

        x = 0.0
        y = 0.0
        position_source = ""
        points: list[tuple[float, float]] = []

        if "inferno" in lowered and fire_count > 0:
            base_addrs = [entity_ptr + self._inferno_fields["m_firePositions"]]
            if inferno_parent_positions is not None:
                base_addrs.append(entity_ptr + inferno_parent_positions)
            for base_addr in base_addrs:
                if points:
                    break
                for flame_idx in range(fire_count):
                    try:
                        px, py, _ = read_vec3(self.pm, base_addr + flame_idx * 12)
                    except Exception:
                        continue
                    if px == 0.0 and py == 0.0:
                        continue
                    points.append((float(px), float(py)))
            if points:
                x = sum(point[0] for point in points) / len(points)
                y = sum(point[1] for point in points) / len(points)
                position_source = "fire_positions"

        if (x == 0.0 and y == 0.0) and "inferno" in lowered:
            if inferno_min_bounds is not None and inferno_max_bounds is not None:
                try:
                    min_x, min_y, _ = read_vec3(self.pm, entity_ptr + inferno_min_bounds)
                    max_x, max_y, _ = read_vec3(self.pm, entity_ptr + inferno_max_bounds)
                    x = float((min_x + max_x) / 2.0)
                    y = float((min_y + max_y) / 2.0)
                    if x != 0.0 or y != 0.0:
                        position_source = "bounds_center"
                except Exception:
                    x = 0.0
                    y = 0.0

        if lowered == "molotov_projectile":
            try:
                x, y, _ = read_vec3(self.pm, entity_ptr + self._grenade_projectile_fields["m_vecExplodeEffectOrigin"])
                x = float(x)
                y = float(y)
                if x != 0.0 or y != 0.0:
                    position_source = "explode_origin"
            except Exception:
                x = 0.0
                y = 0.0

        if lowered == "molotov_projectile" and x == 0.0 and y == 0.0:
            try:
                x, y, _ = read_vec3(self.pm, entity_ptr + self._grenade_projectile_fields["vecLastTrailLinePos"])
                x = float(x)
                y = float(y)
                if x != 0.0 or y != 0.0:
                    position_source = "trail_last"
            except Exception:
                x = 0.0
                y = 0.0

        if x == 0.0 and y == 0.0:
            ox, oy, _ = self._read_entity_origin(entity_ptr)
            x = float(ox)
            y = float(oy)
            if x != 0.0 or y != 0.0:
                position_source = "origin"

        inferno_live_signal = bool(
            is_burning or in_post_effect or fire_count > 0 or tick_begin > 0 or fire_lifetime > 0.0
        )
        projectile_live_signal = bool(
            lowered == "molotov_projectile"
            and (
                explode_effect_began
                or position_source == "explode_origin"
            )
        )
        synthetic_seedable = bool(
            lowered == "molotov_projectile"
            and (x != 0.0 or y != 0.0)
            and (
                explode_effect_began
                or position_source == "explode_origin"
            )
        )
        accepted = bool(
            (("inferno" in lowered) and inferno_live_signal and (x != 0.0 or y != 0.0))
            or (projectile_live_signal and (x != 0.0 or y != 0.0))
        )

        return {
            "class_name": class_name,
            "entity_ptr": int(entity_ptr),
            "is_burning": is_burning,
            "in_post_effect": in_post_effect,
            "explode_effect_began": explode_effect_began,
            "is_live": is_live,
            "is_inc_grenade": is_inc_grenade,
            "bounces": int(bounces),
            "synthetic_seedable": synthetic_seedable,
            "fire_count": int(fire_count),
            "tick_begin": int(tick_begin),
            "fire_lifetime": float(fire_lifetime),
            "inferno_type": int(inferno_type),
            "x": float(x),
            "y": float(y),
            "position_source": position_source or "-",
            "accepted": accepted,
        }

    def debug_inferno_candidates(self) -> list[dict[str, Any]]:
        """Return inferno/molotov-related entity candidates for live debugging."""
        rows: list[dict[str, Any]] = []
        for entity_index, entity_ptr, class_name in self.iterate_entities((1, 8192)):
            info = self._read_inferno_candidate(entity_ptr, class_name)
            if info is None:
                continue
            info = dict(info)
            info["entity_index"] = int(entity_index)
            rows.append(info)
        rows.sort(
            key=lambda item: (
                0 if bool(item.get("accepted")) else 1,
                str(item.get("class_name", "")),
                int(item.get("entity_index", 0)),
            )
        )
        return rows

    def is_attached(self) -> bool:
        """Return True while the process handle and engine module remain readable."""
        try:
            self.pm.read_int(self.engine_base + self._engine_offsets["dwBuildNumber"])
        except Exception:
            return False
        return True

    def read_players(self) -> list[dict[str, Any]]:
        """Read all active T/CT player controllers and their current pawn state."""
        ent_list_base = self._read_entity_list_base()
        if ent_list_base == 0:
            return []

        rows: list[dict[str, Any]] = []
        for entity_index in range(1, MAX_PLAYERS + 1):
            try:
                ctrl = resolve_handle(self.pm, ent_list_base, entity_index)
            except Exception:
                continue
            if ctrl == 0:
                continue

            if entity_class_name(self.pm, ctrl, self.classes) != "cs_player_controller":
                continue

            try:
                pawn_handle = self.pm.read_uint(ctrl + self._controller_fields["m_hPlayerPawn"])
            except Exception:
                continue
            if pawn_handle == _INVALID_HANDLE:
                continue

            try:
                name = read_player_name(self.pm, ctrl, self._base_controller_fields["m_iszPlayerName"])
                if not name:
                    fallback_name_offset = _optional_field(
                        self.classes,
                        "CCSPlayerController",
                        "m_sSanitizedPlayerName",
                    )
                    if fallback_name_offset is not None:
                        name = read_player_name(self.pm, ctrl, fallback_name_offset)

                steam_id = int(self.pm.read_ulonglong(ctrl + self._base_controller_fields["m_steamID"]))
                team_num = int(self.pm.read_uchar(ctrl + self._base_entity_fields["m_iTeamNum"]))
                team_name = TEAM_NAMES.get(team_num, f"T{team_num}")
                if team_name not in {"T", "CT"}:
                    continue

                alive = bool(self.pm.read_uchar(ctrl + self._controller_fields["m_bPawnIsAlive"]))
                hp = int(self.pm.read_int(ctrl + self._controller_fields["m_iPawnHealth"]))
                armor = int(self.pm.read_int(ctrl + self._controller_fields["m_iPawnArmor"]))
                helmet = bool(self.pm.read_uchar(ctrl + self._controller_fields["m_bPawnHasHelmet"]))
                defuser = bool(self.pm.read_uchar(ctrl + self._controller_fields["m_bPawnHasDefuser"]))
                money_services = int(
                    self.pm.read_longlong(ctrl + self._controller_fields["m_pInGameMoneyServices"])
                )
                money = (
                    int(self.pm.read_int(money_services + self._money_fields["m_iAccount"]))
                    if money_services
                    else -1
                )
            except Exception:
                continue

            x = y = z = yaw = 0.0
            scoped = False
            defusing = False
            inventory = self._empty_inventory()
            try:
                pawn = resolve_handle(self.pm, ent_list_base, pawn_handle)
            except Exception:
                pawn = 0

            if pawn:
                inventory = self._read_player_inventory_with_entity_list(pawn, ent_list_base)
                try:
                    x, y, z = self._read_entity_origin(pawn)
                    yaw = float(self.pm.read_float(pawn + self._pawn_fields["m_angEyeAngles"] + 4))
                    scoped = bool(self.pm.read_uchar(pawn + self._pawn_fields["m_bIsScoped"]))
                    defusing = bool(self.pm.read_uchar(pawn + self._pawn_fields["m_bIsDefusing"]))
                except Exception:
                    pass

            rows.append(
                {
                    "idx": entity_index,
                    "steamid": steam_id,
                    "name": name,
                    "team": team_name,
                    "alive": alive,
                    "hp": hp,
                    "armor": armor,
                    "money": money,
                    "helmet": helmet,
                    "defuser": defuser,
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "yaw": float(yaw),
                    "scoped": scoped,
                    "defusing": defusing,
                    **inventory,
                }
            )

        return rows

    def read_player_inventory(self, pawn: int) -> dict[str, Any]:
        """Read inventory handles for one pawn and derive coarse grenade/C4 flags."""
        return self._read_player_inventory_with_entity_list(pawn, self._read_entity_list_base())

    def iterate_entities(
        self,
        index_range: tuple[int, int] = (1, 4096),
    ):
        """Yield (entity_index, entity_ptr, class_name) for populated entity-list slots."""
        ent_list_base = self._read_entity_list_base()
        if ent_list_base == 0:
            return

        for entity_index in range(*index_range):
            try:
                entity_ptr = resolve_handle(self.pm, ent_list_base, entity_index)
            except Exception:
                continue
            if entity_ptr == 0:
                continue

            class_name = entity_class_name(self.pm, entity_ptr, self.classes)
            if class_name:
                yield entity_index, entity_ptr, class_name

    def read_map_state(self) -> dict[str, Any]:
        """Best-effort read of live round state from C_CSGameRules and team entities."""
        state: dict[str, Any] = {
            "map_name": "",
            "round_num": 0,
            "ct_score": 0,
            "t_score": 0,
            "time_in_round": 0.0,
            "map_phase": "live",
            "round_phase": "live",
            "bomb_state": "",
            "bomb": {"planted": False, "dropped": False, "x": 0.0, "y": 0.0, "site": ""},
            "projectiles": {"smokes": [], "molotovs": []},
        }

        ent_list_base = self._read_entity_list_base()

        rules_ptr = self._resolve_game_rules_ptr()
        if rules_ptr == 0:
            if ent_list_base:
                state["ct_score"], state["t_score"] = self._read_team_scores(ent_list_base)
            state["bomb"] = self.read_bomb()
            state["projectiles"] = self.read_projectiles()
            return state

        # Prefer the gamerules RoundResults array (works in demo playback
        # where C_CSTeam entities don't appear in the entity list). Fall back
        # to the entity-list scan only if the gamerules read yields 0,0.
        ct_score, t_score = self._read_match_scores(rules_ptr)
        if ct_score == 0 and t_score == 0 and ent_list_base:
            ct_score, t_score = self._read_team_scores(ent_list_base)
        state["ct_score"] = ct_score
        state["t_score"] = t_score

        total_rounds = self._read_optional_int(
            rules_ptr,
            "C_CSGameRules",
            "m_totalRoundsPlayed",
            warn_key="C_CSGameRules.m_totalRoundsPlayed",
            default=0,
        )
        state["round_num"] = max(0, int(total_rounds or 0) + 1)

        warmup = self._read_optional_bool(
            rules_ptr,
            "C_CSGameRules",
            "m_bWarmupPeriod",
            warn_key="C_CSGameRules.m_bWarmupPeriod",
            default=False,
        )
        freeze = self._read_optional_bool(
            rules_ptr,
            "C_CSGameRules",
            "m_bFreezePeriod",
            warn_key="C_CSGameRules.m_bFreezePeriod",
            default=False,
        )
        bomb_planted = self._read_optional_bool(
            rules_ptr,
            "C_CSGameRules",
            "m_bBombPlanted",
            warn_key="C_CSGameRules.m_bBombPlanted",
            default=False,
        )
        has_match_started = self._read_optional_bool(
            rules_ptr,
            "C_CSGameRules",
            "m_bHasMatchStarted",
            warn_key="C_CSGameRules.m_bHasMatchStarted",
            default=not warmup,
        )

        state["map_phase"] = "warmup" if warmup or not has_match_started else "live"
        state["round_phase"] = "freezetime" if freeze else "live"
        state["bomb_state"] = "planted" if bomb_planted else ""
        state["map_name"] = self._read_map_name(rules_ptr)

        round_time = self._read_optional_int(
            rules_ptr,
            "C_CSGameRules",
            "m_iRoundTime",
            warn_key="C_CSGameRules.m_iRoundTime",
            default=0,
        )
        time_to_next = self._read_optional_float(
            rules_ptr,
            "C_CSGameRules",
            "m_timeUntilNextPhaseStarts",
            warn_key="C_CSGameRules.m_timeUntilNextPhaseStarts",
            default=None,
        )
        if time_to_next is not None and round_time and not warmup and not freeze:
            elapsed = float(round_time) - max(0.0, float(time_to_next))
            state["time_in_round"] = max(0.0, min(float(round_time), elapsed))
        else:
            round_start_field = _optional_field(self.classes, "C_CSGameRules", "m_fRoundStartTime")
            if round_start_field is None:
                round_start_field = _optional_field(self.classes, "C_CSGameRules", "m_flRoundStartTime")
            if round_start_field is None:
                self._warn_once(
                    "C_CSGameRules.round_start_time",
                    "client_dll.json missing C_CSGameRules.m_fRoundStartTime/m_flRoundStartTime; time_in_round will default to 0",
                )

        state["bomb"] = self.read_bomb()
        state["projectiles"] = self.read_projectiles()
        return state

    def read_bomb(self) -> dict[str, Any]:
        """Read planted or dropped C4 state from the verified client.dll vectors."""
        bomb_state = {
            "dropped": False,
            "x": 0.0,
            "y": 0.0,
            "planted": False,
            "site": "",
        }
        planted_size, planted_ptr = self._read_utl_vector(
            self.client_base + self._client_offsets["dwPlantedC4"]
        )
        for slot in range(min(planted_size, 8)):
            try:
                c4_ptr = int(self.pm.read_longlong(planted_ptr + 8 * slot))
            except Exception:
                continue
            if c4_ptr == 0:
                continue

            try:
                bomb_ticking = bool(
                    self.pm.read_uchar(c4_ptr + self._planted_c4_fields["m_bBombTicking"])
                )
                bomb_defused = bool(
                    self.pm.read_uchar(c4_ptr + self._planted_c4_fields["m_bBombDefused"])
                )
                bomb_site = int(self.pm.read_int(c4_ptr + self._planted_c4_fields["m_nBombSite"]))
            except Exception:
                continue
            if not bomb_ticking or bomb_defused:
                continue

            x, y, _ = self._read_entity_origin(c4_ptr)
            bomb_state["planted"] = True
            bomb_state["dropped"] = False
            bomb_state["x"] = float(x)
            bomb_state["y"] = float(y)
            bomb_state["site"] = "A" if bomb_site == 0 else "B" if bomb_site == 1 else ""
            return bomb_state

        ent_list_base = self._read_entity_list_base()
        if ent_list_base == 0:
            return bomb_state

        c4_size, c4_handles = self._read_utl_vector(self.client_base + self._client_offsets["dwWeaponC4"])
        for slot in range(min(c4_size, _MAX_INVENTORY_WEAPONS)):
            try:
                c4_handle = int(self.pm.read_uint(c4_handles + 4 * slot))
            except Exception:
                continue
            c4_ptr = resolve_handle(self.pm, ent_list_base, c4_handle)
            if c4_ptr == 0:
                continue

            try:
                owner = int(self.pm.read_uint(c4_ptr + self._base_entity_fields["m_hOwnerEntity"]))
            except Exception:
                continue
            if owner not in (0, _INVALID_HANDLE):
                continue

            x, y, _ = self._read_entity_origin(c4_ptr)
            bomb_state["dropped"] = True
            bomb_state["x"] = float(x)
            bomb_state["y"] = float(y)
            return bomb_state

        return bomb_state

    def read_projectiles(self) -> dict[str, Any]:
        """Read active smoke and inferno entities from the full entity list."""
        smokes: list[tuple[float, float, float]] = []
        molotovs: list[tuple[float, float, float]] = []
        seen_smokes: set[int] = set()
        seen_infernos: set[int] = set()
        seen_molotov_projectiles: set[int] = set()

        for _, entity_ptr, class_name in self.iterate_entities():
            if class_name == "smokegrenade_projectile":
                seen_smokes.add(entity_ptr)
                try:
                    did_smoke = bool(
                        self.pm.read_uchar(entity_ptr + self._smoke_fields["m_bDidSmokeEffect"])
                    )
                    tick_begin = int(
                        self.pm.read_int(entity_ptr + self._smoke_fields["m_nSmokeEffectTickBegin"])
                    )
                except Exception:
                    continue
                if not did_smoke:
                    continue

                try:
                    x, y, _ = read_vec3(
                        self.pm,
                        entity_ptr + self._smoke_fields["m_vSmokeDetonationPos"],
                    )
                except Exception:
                    x, y, _ = self._read_entity_origin(entity_ptr)
                if x == 0.0 and y == 0.0:
                    x, y, _ = self._read_entity_origin(entity_ptr)

                remain = self._projectile_remain(
                    self._smoke_seen_at,
                    entity_ptr,
                    tick_begin,
                    _SMOKE_DURATION_SECONDS,
                )
                if remain > 0.0:
                    smokes.append((float(x), float(y), remain))
                continue

            info = self._read_inferno_candidate(entity_ptr, class_name)
            if info is None:
                continue
            lowered = str(info.get("class_name", "")).lower()
            if lowered not in {"inferno", "molotov_projectile"}:
                continue
            seen_infernos.add(entity_ptr)
            if lowered == "molotov_projectile":
                seen_molotov_projectiles.add(entity_ptr)
                prior_state = self._molotov_projectile_state.get(entity_ptr)
                current_x = float(info.get("x", 0.0))
                current_y = float(info.get("y", 0.0))
                current_live = bool(info.get("is_live"))
                current_source = str(info.get("position_source", ""))
                current_bounces = int(info.get("bounces", 0))
                observed_frames = int(prior_state.get("frames", 0)) + 1 if prior_state else 1

                should_seed_transition = bool(
                    prior_state
                    and bool(prior_state.get("is_live"))
                    and not current_live
                    and (current_x != 0.0 or current_y != 0.0)
                    and (current_source in {"explode_origin", "trail_last"} or current_bounces > 0)
                )
                should_seed_explicit = bool(info.get("synthetic_seedable"))
                if should_seed_transition or should_seed_explicit:
                    self._seed_synthetic_molotov(
                        x=current_x,
                        y=current_y,
                        duration_seconds=(
                            float(info.get("fire_lifetime", 0.0))
                            if float(info.get("fire_lifetime", 0.0)) > 0.0
                            else _MOLOTOV_DURATION_SECONDS
                        ),
                        signature=int(entity_ptr),
                    )

                self._molotov_projectile_state[entity_ptr] = {
                    "x": current_x,
                    "y": current_y,
                    "is_live": current_live,
                    "position_source": str(info.get("position_source", "")),
                    "bounces": int(info.get("bounces", 0)),
                    "frames": observed_frames,
                    "last_seen_at": time.monotonic(),
                }

            if bool(info.get("synthetic_seedable")) and lowered == "inferno":
                self._seed_synthetic_molotov(
                    x=float(info.get("x", 0.0)),
                    y=float(info.get("y", 0.0)),
                    duration_seconds=(
                        float(info.get("fire_lifetime", 0.0))
                        if float(info.get("fire_lifetime", 0.0)) > 0.0
                        else _MOLOTOV_DURATION_SECONDS
                    ),
                    signature=int(info.get("tick_begin", 0)) or int(entity_ptr),
                )
            if not bool(info.get("accepted")):
                continue

            remain = self._projectile_remain(
                self._inferno_seen_at,
                entity_ptr,
                int(info.get("tick_begin", 0)),
                (
                    float(info.get("fire_lifetime", 0.0))
                    if float(info.get("fire_lifetime", 0.0)) > 0.0
                    else _MOLOTOV_DURATION_SECONDS
                ),
            )
            if remain > 0.0:
                molotovs.append((float(info.get("x", 0.0)), float(info.get("y", 0.0)), remain))

        self._trim_projectile_cache(self._smoke_seen_at, seen_smokes)
        self._trim_projectile_cache(self._inferno_seen_at, seen_infernos)
        stale_projectiles = [
            entity_ptr
            for entity_ptr in self._molotov_projectile_state
            if entity_ptr not in seen_molotov_projectiles
        ]
        for entity_ptr in stale_projectiles:
            state = self._molotov_projectile_state.pop(entity_ptr)
            observed_frames = int(state.get("frames", 0))
            x = float(state.get("x", 0.0))
            y = float(state.get("y", 0.0))
            position_source = str(state.get("position_source", ""))
            bounces = int(state.get("bounces", 0))
            if (
                observed_frames >= 2
                and (x != 0.0 or y != 0.0)
                and (position_source in {"explode_origin", "trail_last"} or bounces > 0)
            ):
                self._seed_synthetic_molotov(
                    x=x,
                    y=y,
                    duration_seconds=_MOLOTOV_DURATION_SECONDS,
                    signature=int(entity_ptr),
                )
        for x, y, remain in self._active_synthetic_molotovs():
            if any(abs(existing_x - x) <= 96.0 and abs(existing_y - y) <= 96.0 for existing_x, existing_y, _ in molotovs):
                continue
            molotovs.append((x, y, remain))
        smokes.sort(key=lambda item: (-item[2], item[0], item[1]))
        molotovs.sort(key=lambda item: (-item[2], item[0], item[1]))
        return {
            "smokes": smokes[:5],
            "molotovs": molotovs[:3],
        }
