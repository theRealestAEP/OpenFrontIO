import path from "node:path";
import { performance } from "node:perf_hooks";
import quickChatData from "resources/QuickChat.json";
import { Config } from "../core/configuration/Config";
import { Executor } from "../core/execution/ExecutionManager";
import { GameRunner } from "../core/GameRunner";
import {
  AllPlayers,
  Attack,
  Difficulty,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  Player,
  PlayerInfo,
  PlayerBuildable,
  PlayerBuildableUnitType,
  PlayerID,
  PlayerType,
  TerraNullius,
  TeamGameSpawnAreas,
  TerrainType,
  UnitType,
} from "../core/game/Game";
import { TileRef } from "../core/game/GameMap";
import { ErrorUpdate, GameUpdateViewData } from "../core/game/GameUpdates";
import { createGame } from "../core/game/GameImpl";
import { createNationsForGame } from "../core/game/NationCreation";
import { genTerrainFromBin } from "../core/game/TerrainMapLoader";
import { PseudoRandom } from "../core/PseudoRandom";
import {
  ClientID,
  GameRecord,
  GameConfig,
  GameStartInfo,
  Intent,
  PlayerRecord,
  StampedIntent,
  Turn,
  Winner,
} from "../core/Schemas";
import { createPartialGameRecord, simpleHash } from "../core/Util";
import { FileMapLoader } from "./FileMapLoader";

type CandidateKind = "noop" | "intent";
type ActionProfile = "full" | "bot" | "bot-lite";
type RawActionType = "noop" | "spawn" | "attack" | "boat" | "build_unit";
type CandidateAdder = ((candidate: ActionCandidate) => void) & {
  hasCapacity: () => boolean;
};

export interface HeadlessEpisodeConfig {
  map: GameMapType;
  difficulty: Difficulty;
  seed: string;
  actionProfile: ActionProfile;
  boatDestinationProbeLimit: number | "full";
  boatCandidateInterval: number;
  buildProbeLimit: number | "full";
  buildCandidateInterval: number;
  localFeatureRadii: number[];
  bots: number;
  nations: number | "default" | "disabled";
  // Number of ML-controlled human players in the lobby. 1 = the classic
  // single-agent path (unchanged). >1 enables multi-agent self-play: the
  // remaining slots are driven by `stepMulti`, each with its own egocentric
  // observation, and the rest of the lobby is filled by bots/nations.
  controlledPlayers: number;
  mapsRoot: string;
  decisionInterval: number;
  maxTurns: number;
  spatialSize: number;
  maxActions: number;
  maxPlayers: number;
  compactCandidates: boolean;
  compactSpatial: boolean;
  // Curriculum knob: override the % of map tiles needed to win (engine default
  // 80 for FFA). Lower it so RL registers real wins at the agent's current
  // capability, then ladder it back up to 80. Undefined = engine default.
  winPercent?: number;
}

export interface ActionCandidate {
  kind: CandidateKind;
  label: string;
  intent?: Intent;
  features: number[];
}

export interface ObservationActionCandidate {
  kind?: CandidateKind;
  label?: string;
  intent?: Intent;
  features: number[];
}

export interface HeadlessObservation {
  tick: number;
  turnNumber: number;
  spatialSize: number;
  spatial: Record<string, number[]>;
  spatialBinary?: {
    order: string[];
    dtype: "float32" | "uint8";
    size: number;
    data: string;
  };
  vector: number[];
  players: PlayerSummary[];
  candidates: ObservationActionCandidate[];
  candidateCount: number;
  candidateXYIndex: number;
}

export interface PlayerSummary {
  id: PlayerID;
  smallID: number;
  type: PlayerType;
  isSelf: boolean;
  isAlive: boolean;
  tiles: number;
  troops: number;
  gold: number;
  outgoingAttacks: number;
  incomingAttacks: number;
  units: Partial<Record<UnitType, number>>;
}

export interface HeadlessStepResult {
  observation: HeadlessObservation;
  reward: number;
  done: boolean;
  info: Record<string, unknown>;
}

export interface HeadlessMultiStepResult {
  observations: HeadlessObservation[];
  rewards: number[];
  dones: boolean[];
  done: boolean;
  info: Record<string, unknown>;
}

export interface RawActionSlot {
  type?: RawActionType;
  tile?: number;
  x?: number;
  y?: number;
  troopRatio?: number;
  targetID?: PlayerID | null;
  targetIndex?: number;
  unit?: UnitType;
  rocketDirectionUp?: boolean;
}

interface RawIntentResult {
  intent?: Intent;
  reason?: string;
}

interface Metrics {
  alive: boolean;
  cities: number;
  enemiesAlive: number;
  gold: number;
  myShare: number;
  myTiles: number;
  ports: number;
  rank: number;
  troops: number;
  // Tech: stable structure counts plus in-flight nukes (a transient positive
  // delta on launch). Small reward nudges so the policy learns these tools
  // exist; the win/land signal stays dominant.
  missileSilos: number;
  sams: number;
  nukesInFlight: number;
}

interface OpponentCounts {
  bots: number;
  nations: number;
}

interface SpatialCache {
  size: number;
  numLandTiles: number;
  cellByTile: number[];
  denom: number[];
  landCounts: number[];
  landTiles: TileRef[];
  staticPlanes: {
    land: number[];
    shore: number[];
  };
}

// Stage-2 "hybrid retina": a fixed-resolution 64x64 downsample of the full tile
// grid, produced ONLY when observeEntities is asked for it (opt-in). Separate
// from spatialCache so it never thrashes the size-32 observe() cache when both
// are computed on the same step. Static (terrain) planes are precomputed once
// per (size, numLandTiles); dynamic planes are rebuilt per call.
const RETINA_SIZE = 64;
// Plane order is the model's channel order — APPEND-ONLY, never reorder (v2
// checkpoints are channel-aligned like the action vocabulary).
const RETINA_PLANES = [
  "own", // 0 own territory presence
  "enemy", // 1 enemy territory intensity (per-cell enemy land share)
  "neutral", // 2 neutral (unowned) land presence
  "water", // 3 water presence
  "mountain", // 4 mountain / impassable presence
  "fallout", // 5 fallout presence
  "ownBorderUnderAttack", // 6 own border tile adjacent to enemy-owned land
  "enemyStructure", // 7 enemy structure density
] as const;
const RETINA_STRUCT_NORM = 3; // enemy-structure count that saturates the plane

interface RetinaCache {
  size: number;
  numLandTiles: number;
  cellByTile: number[];
  landTileCount: number[]; // land tiles per cell (enemy-share denominator)
  landTiles: TileRef[];
  landPresence: Uint8Array; // static: any land tile in cell
  waterPresence: Uint8Array; // static: any water tile in cell
  mountainPresence: Uint8Array; // static: any mountain tile in cell
}

interface CandidateFeatureContext {
  actor: Player | null;
  actorFeatures: number[];
  leaderID: PlayerID | null;
  playerCount: number;
  rankByID: Map<PlayerID, number>;
  targetFeatureCache: Map<PlayerID, number[]>;
  tileFeatureCache: Map<TileRef, number[]>;
  localPatchFeatureCache: Map<string, number[]>;
  nearestUnitCache: Map<string, number>;
  nearbyUnitCountCache: Map<string, number>;
  borderContactCache: Map<PlayerID, number>;
}

const DEFAULT_CONFIG: HeadlessEpisodeConfig = {
  map: GameMapType.World,
  difficulty: Difficulty.Easy,
  seed: "OPENAI001",
  actionProfile: "full",
  boatDestinationProbeLimit: "full",
  boatCandidateInterval: 1,
  buildProbeLimit: 80,
  buildCandidateInterval: 1,
  localFeatureRadii: [],
  bots: 0,
  nations: "default",
  controlledPlayers: 1,
  mapsRoot: path.resolve("resources", "maps"),
  decisionInterval: 10,
  maxTurns: 18_000,
  spatialSize: 32,
  maxActions: 256,
  maxPlayers: 64,
  compactCandidates: false,
  compactSpatial: false,
};

const CLIENT_ID = "MLAGENT1";
const USERNAME = "OpenFrontAI";
// Deterministic client IDs for the N controlled players. Index 0 is the
// "hero" (== CLIENT_ID, preserving the single-agent identity); the rest are
// the self-play opponent slots.
function controlledClientID(index: number): ClientID {
  return index === 0 ? CLIENT_ID : `MLAGENT${index + 1}`;
}
const MAX_OPPONENT_PRESPAWN_TURNS = 512;
const MAX_RAW_INTENTS_PER_STEP = 8;
const TROOP_RATIOS = [0.15, 0.3, 0.5, 0.75, 1.0] as const;
const EMOJIS = [0, 1, 13, 15, 33] as const;
const ACTION_TYPE_NAMES = [
  "noop",
  "spawn",
  "attack",
  "boat",
  "build_unit",
  "upgrade_structure",
  "cancel_attack",
  "cancel_boat",
  "move_warship",
  "allianceRequest",
  "allianceReject",
  "allianceExtension",
  "breakAlliance",
  "targetPlayer",
  "emoji",
  "donate_gold",
  "donate_troops",
  "embargo",
  "embargo_all",
  "quick_chat",
  "delete_unit",
] as const;
const ACTION_UNIT_NAMES = [
  UnitType.TransportShip,
  UnitType.Warship,
  UnitType.Port,
  UnitType.City,
  UnitType.MissileSilo,
  UnitType.DefensePost,
  UnitType.SAMLauncher,
  UnitType.Factory,
  UnitType.AtomBomb,
  UnitType.HydrogenBomb,
  UnitType.MIRV,
] as const;
const BOT_LITE_BUILD_UNITS = new Set<UnitType>([
  UnitType.City,
  UnitType.Port,
  UnitType.DefensePost,
]);
const LAND_STRUCTURE_BUILD_UNITS: PlayerBuildableUnitType[] = [
  UnitType.City,
  UnitType.DefensePost,
  UnitType.SAMLauncher,
  UnitType.MissileSilo,
  UnitType.Factory,
];
const SPECIAL_BUILD_UNITS: PlayerBuildableUnitType[] = [
  UnitType.Port,
  UnitType.AtomBomb,
  UnitType.HydrogenBomb,
  UnitType.MIRV,
  UnitType.Warship,
];
const QUICK_CHAT_KEYS = Object.entries(quickChatData)
  .flatMap(([category, entries]) =>
    entries.map((entry) => ({
      key: `${category}.${entry.key}`,
      requiresPlayer: entry.requiresPlayer,
    })),
  )
  .slice(0, 12);

export class OpenFrontGymEnv {
  private config: HeadlessEpisodeConfig = DEFAULT_CONFIG;
  private runner: GameRunner | null = null;
  private clientID: ClientID = CLIENT_ID;
  // All ML-controlled client IDs (hero first). Single-agent runs hold just
  // [CLIENT_ID]; multi-agent self-play holds N entries.
  private controlledClientIDs: ClientID[] = [CLIENT_ID];
  // Per-controlled-player previous metrics + cached candidate lists, used by
  // the multi-agent step path so each slot gets its own reward + action map.
  private previousMetricsByClient: Map<ClientID, Metrics> = new Map();
  private lastCandidatesByClient: Map<ClientID, ActionCandidate[]> = new Map();
  private turnNumber = 0;
  private decisionNumber = 0;
  private gameStartInfo: GameStartInfo | null = null;
  private turns: Turn[] = [];
  private startedAt = 0;
  private lastError: ErrorUpdate | null = null;
  private lastUpdate: GameUpdateViewData | null = null;
  private previousMetrics: Metrics | null = null;
  private lastTimings: Record<string, number> = {};
  private lastCandidates: ActionCandidate[] = [];
  private maxTurnNumber = DEFAULT_CONFIG.maxTurns;
  private spatialCache: SpatialCache | null = null;
  private retinaCache: RetinaCache | null = null;
  private shoreLandTilesCache: TileRef[] | null = null;
  private candidateTimings: Record<string, number> | null = null;
  private candidateFeatureContext: CandidateFeatureContext | null = null;
  private actionTranscript: Array<{ turn: number; candidate: ActionCandidate }> =
    [];

  async reset(
    config: Partial<HeadlessEpisodeConfig> = {},
  ): Promise<HeadlessStepResult> {
    const started = performance.now();
    this.config = { ...DEFAULT_CONFIG, ...config };
    const numControlled = Math.max(1, Math.floor(this.config.controlledPlayers));
    this.controlledClientIDs = Array.from({ length: numControlled }, (_, i) =>
      controlledClientID(i),
    );
    this.clientID = this.controlledClientIDs[0];
    this.turnNumber = 0;
    this.decisionNumber = 0;
    this.maxTurnNumber = this.config.maxTurns;
    this.turns = [];
    this.lastError = null;
    this.lastUpdate = null;
    this.lastCandidates = [];
    this.lastCandidatesByClient = new Map();
    this.previousMetricsByClient = new Map();
    this.spatialCache = null;
    this.retinaCache = null;
    this.shoreLandTilesCache = null;
    this.actionTranscript = [];
    this.startedAt = Date.now();

    const gameStart = buildGameStartInfo(this.config, this.controlledClientIDs);
    this.gameStartInfo = gameStart;
    this.runner = await createIsolatedGameRunner(
      gameStart,
      this.clientID,
      this.config.mapsRoot,
      (update) => {
        if ("errMsg" in update) {
          this.lastError = update;
        } else {
          this.lastUpdate = update;
        }
      },
      this.config.winPercent,
    );
    const preSpawnStarted = performance.now();
    const preSpawnTurns = this.preSpawnOpponents();
    const preSpawnMs = performance.now() - preSpawnStarted;
    this.maxTurnNumber = this.turnNumber + this.config.maxTurns;
    this.previousMetrics = this.metrics();
    for (const cid of this.controlledClientIDs) {
      this.previousMetricsByClient.set(cid, this.metrics(cid));
    }
    return this.result(
      0,
      {
        preSpawnMs,
        preSpawnTurns,
        resetMs: performance.now() - started,
      },
      started,
    );
  }

  step(actionIndex: number): HeadlessStepResult {
    const started = performance.now();
    this.requireRunner();
    const candidateStarted = performance.now();
    const candidates =
      this.lastCandidates.length > 0 ? this.lastCandidates : this.generateCandidates();
    const actionCandidateMs = performance.now() - candidateStarted;
    if (!Number.isInteger(actionIndex) || actionIndex < 0) {
      throw new Error(`invalid actionIndex ${actionIndex}`);
    }
    const candidate = candidates[actionIndex] ?? candidates[0];
    this.actionTranscript.push({ turn: this.turnNumber, candidate });
    const stamped =
      candidate.kind === "intent" && candidate.intent !== undefined
        ? ({ ...candidate.intent, clientID: this.clientID } as StampedIntent)
        : undefined;

    return this.advanceWithIntents(
      stamped !== undefined ? [stamped] : [],
      started,
      { actionCandidateMs },
    );
  }

  /**
   * One egocentric observation per controlled player (hero first). Caches each
   * player's candidate list so a subsequent `stepMulti` can map its action
   * index back to an intent. Used once after `reset` to seed the self-play
   * loop, and is also returned by every `stepMulti`.
   */
  observeAll(): HeadlessObservation[] {
    this.requireRunner();
    return this.controlledClientIDs.map((cid) => this.observe(cid).observation);
  }

  /**
   * Reconstruct the EXACT production game from a replay record's start info
   * (real players, no injected MLAGENT slots, no pre-spawn). The runner seeds
   * its RNG from `info.gameID`, so replaying the recorded intent stream via
   * `stepIntents` reproduces the production game deterministically. Used by
   * replay ingestion to build behavior-cloning data.
   */
  async resetFromRecord(
    info: GameStartInfo,
    observeClientIDs?: ClientID[],
    mapsRoot?: string,
    spatialSize?: number,
  ): Promise<void> {
    // Replay records have ONE entry per game tick, so each step_intents must
    // advance exactly one tick (not the training decisionInterval) or the
    // simulation desyncs from the recorded game. spatialSize overrides the
    // observation resolution (raw-intent BC uses high res — no per-step cost).
    this.config = {
      ...DEFAULT_CONFIG,
      decisionInterval: 1,
      ...(mapsRoot ? { mapsRoot } : {}),
      ...(spatialSize ? { spatialSize } : {}),
    };
    this.turnNumber = 0;
    this.decisionNumber = 0;
    this.maxTurnNumber = Number.MAX_SAFE_INTEGER;
    this.turns = [];
    this.lastError = null;
    this.lastUpdate = null;
    this.lastCandidates = [];
    this.lastCandidatesByClient = new Map();
    this.previousMetricsByClient = new Map();
    this.spatialCache = null;
    this.retinaCache = null;
    this.shoreLandTilesCache = null;
    this.actionTranscript = [];
    this.startedAt = Date.now();
    this.gameStartInfo = info;
    this.controlledClientIDs =
      observeClientIDs && observeClientIDs.length > 0
        ? observeClientIDs
        : info.players.map((player) => player.clientID);
    this.clientID = this.controlledClientIDs[0];
    this.runner = await createIsolatedGameRunner(
      info,
      this.clientID,
      this.config.mapsRoot,
      (update) => {
        if ("errMsg" in update) {
          this.lastError = update;
        } else {
          this.lastUpdate = update;
        }
      },
    );
    this.previousMetrics = this.metrics();
  }

  /**
   * Egocentric observation + candidate list for each requested client WITHOUT
   * advancing the game. Used by replay ingestion to label a cloned player's
   * recorded intent against our generated candidates at each decision turn.
   */
  observeClients(
    clientIDs: ClientID[],
  ): Record<string, { observation: HeadlessObservation }> {
    this.requireRunner();
    const out: Record<string, { observation: HeadlessObservation }> = {};
    for (const cid of clientIDs) {
      out[cid] = { observation: this.observe(cid).observation };
    }
    return out;
  }

  /**
   * Replay primitive: advance `ticks` simulation ticks with no intents. Record
   * turn entries are SPARSE (turnNumber is the game tick; empty ticks omitted),
   * so the caller fills the gap before each entry to stay in lockstep.
   */
  replayAdvance(ticks: number): void {
    this.requireRunner();
    for (let i = 0; i < ticks; i++) {
      if (this.isDone()) break;
      this.enqueueTurn([]);
      this.runner!.executeNextTick();
      if (this.lastError !== null) throw new Error(this.lastError.errMsg);
    }
  }

  /**
   * Replay primitive: apply one recorded turn's intents on exactly one tick.
   * Record intents are already client-stamped, so they route to their players.
   */
  replayApply(intents: StampedIntent[]): void {
    this.requireRunner();
    if (this.isDone()) return;
    this.enqueueTurn(intents);
    this.runner!.executeNextTick();
    if (this.lastError !== null) throw new Error(this.lastError.errMsg);
  }

  /**
   * RAW-INTENT BC: decode tile refs to normalized (x,y) in [0,1]. These are the
   * labels for the policy's spatial (heatmap) action head — full map precision,
   * no candidate generator.
   */
  decodeTiles(tiles: number[]): Array<{ x: number; y: number }> {
    const game = this.requireRunner().game;
    const w = Math.max(1, game.width());
    const h = Math.max(1, game.height());
    return tiles.map((t) => ({
      x: game.x(t as TileRef) / w,
      y: game.y(t as TileRef) / h,
    }));
  }

  /** A VALID spawn tile for `cid` (from the engine's own spawn candidates).
   * Self-play eval needs the agent to spawn validly or the game stays stuck in
   * the spawn phase; the BC policy's spawn placement is undertrained, so we use
   * the engine's legal spawn here and let the policy play once established. */
  /** Live outgoing attacks for `cid` — cancel_attack intents need an attackID,
   * which the observation tokens don't carry (attacks are not units). */
  outgoingAttacks(cid: ClientID = this.clientID): {
    id: string;
    troops: number;
    targetID: string | null;
  }[] {
    const game = this.requireRunner().game;
    const me = game.playerByClientID(cid);
    if (me === null) return [];
    return me.outgoingAttacks().map((a) => ({
      id: a.id(),
      troops: a.troops(),
      targetID: a.target().isPlayer() ? (a.target() as Player).id() : null,
    }));
  }

  validSpawnTile(cid: ClientID = this.clientID): number | null {
    const candidates = this.generateCandidates(cid);
    for (const c of candidates) {
      if (c.kind === "intent" && c.intent !== undefined &&
          (c.intent as { type?: string }).type === "spawn") {
        const tile = (c.intent as { tile?: number }).tile;
        if (tile !== undefined) return tile;
      }
    }
    return null;
  }

  /** Inverse of decodeTiles: normalized (x,y) in [0,1] -> nearest valid TileRef.
   * Lets the entity policy's continuous-coordinate action head emit an exact
   * tile to act on. */
  encodeTile(x: number, y: number): number {
    const game = this.requireRunner().game;
    const w = game.width();
    const h = game.height();
    const cx = Math.min(w - 1, Math.max(0, Math.round(x * w)));
    const cy = Math.min(h - 1, Math.max(0, Math.round(y * h)));
    return game.ref(cx, cy) as number;
  }

  /**
   * RAW-INTENT BC: the observation WITHOUT generating candidates (planes +
   * global vector only). The policy emits raw intents directly, so candidate
   * generation — the bottleneck — is dropped from the data path entirely.
   */
  observeRaw(cid: ClientID = this.clientID): HeadlessObservation {
    this.requireRunner();
    const full = this.observe(cid).observation;
    return {
      ...full,
      candidates: [],
      candidateCount: 0,
    };
  }

  /**
   * ENTITY-TOKEN observation — sparse, EXACT positions, NO downsampled grid.
   * Every structure/army is a token with its precise normalized (x,y) and
   * attributes; the policy attends over these and targets them by pointer or
   * exact coordinate. This is how we keep full map resolution without a dense
   * 1500x1500 tensor.
   */
  observeEntities(
    cid: ClientID = this.clientID,
    opts: { spatial2?: boolean } = {},
  ): {
    vector: number[];
    players: PlayerSummary[];
    tokens: Array<{
      kind: number; owner: number; rel: number;
      x: number; y: number; troops: number; health: number;
    }>;
    spatial2?: {
      planes: string[];
      size: number;
      dtype: "uint8";
      data: string;
    };
  } {
    const game = this.requireRunner().game;
    const me = game.playerByClientID(cid);
    const w = Math.max(1, game.width());
    const h = Math.max(1, game.height());
    const TOKEN_TYPES = [
      UnitType.City, UnitType.Port, UnitType.MissileSilo, UnitType.SAMLauncher,
      UnitType.DefensePost, UnitType.Factory, UnitType.Warship,
      UnitType.AtomBomb, UnitType.HydrogenBomb, UnitType.MIRV,
      UnitType.TransportShip,
    ];
    const kindIndex = new Map(TOKEN_TYPES.map((t, i) => [t, i]));
    const tokens = [];
    for (const u of game.units(...TOKEN_TYPES)) {
      if (!u.isActive()) continue;
      const tile = u.tile();
      const owner = u.owner();
      const isMe = me !== null && owner === me;
      tokens.push({
        kind: kindIndex.get(u.type()) ?? 0,
        owner: owner.smallID(),
        rel: isMe ? 0 : owner.isPlayer() ? 2 : 1, // self / nation-bot / player
        x: game.x(tile) / w,
        y: game.y(tile) / h,
        troops: Math.log1p(Math.max(0, u.troops?.() ?? 0)),
        health: Math.log1p(Math.max(0, Number(u.health?.() ?? 0))),
        // id/unitType are NOT part of the model observation (the encoder reads
        // only the fields above) — they let the intent translator resolve
        // "policy coordinate -> concrete unit" for move_warship / upgrade /
        // cancel / delete intents, which the engine validates by unit id.
        id: u.id(),
        unitType: u.type(),
      });
    }
    const obs = this.observe(cid).observation;
    const result: {
      vector: number[];
      players: PlayerSummary[];
      tokens: typeof tokens;
      spatial2?: {
        planes: string[];
        size: number;
        dtype: "uint8";
        data: string;
      };
    } = { vector: obs.vector, players: obs.players, tokens };
    if (opts.spatial2) {
      result.spatial2 = this.retinaPlanes(cid);
    }
    return result;
  }

  /**
   * Stage-2 hybrid retina: 8 downsampled 64x64 uint8 planes (see RETINA_PLANES)
   * packed plane-major, row-major, base64-encoded. Presence planes are max-pool
   * over the tile grid (255 if any qualifying tile lands in the cell); the two
   * graded planes (enemy territory, enemy structures) carry a per-cell intensity.
   * Opt-in via observeEntities({ spatial2: true }) so v1 consumers are untouched.
   */
  private retinaPlanes(cid: ClientID = this.clientID): {
    planes: string[];
    size: number;
    dtype: "uint8";
    data: string;
  } {
    const game = this.requireRunner().game;
    const me = game.playerByClientID(cid);
    const size = RETINA_SIZE;
    const cache = this.getRetinaCache(size);
    const cellCount = size * size;
    const cellByTile = cache.cellByTile;

    // Dynamic accumulators (per cell).
    const ownCount = new Int32Array(cellCount);
    const enemyLandCount = new Int32Array(cellCount);
    const falloutPresence = new Uint8Array(cellCount);
    const borderUnderAttack = new Uint8Array(cellCount);
    const enemyStructCount = new Int32Array(cellCount);

    for (const owner of game.players()) {
      const isSelf = owner.id() === me?.id();
      for (const tile of owner.tiles()) {
        const i = cellByTile[tile];
        if (i === undefined) continue;
        if (isSelf) ownCount[i]++;
        else enemyLandCount[i]++;
      }
    }

    // Own border tiles that neighbor enemy-owned land — the "under attack" front.
    if (me !== null) {
      for (const tile of me.borderTiles()) {
        const i = cellByTile[tile];
        if (i === undefined) continue;
        for (const n of game.neighbors(tile)) {
          const nOwner = game.owner(n);
          if (nOwner.isPlayer() && nOwner !== me) {
            borderUnderAttack[i] = 1;
            break;
          }
        }
      }
    }

    if (game.numTilesWithFallout() > 0) {
      for (const tile of cache.landTiles) {
        if (!game.hasFallout(tile)) continue;
        const i = cellByTile[tile];
        if (i !== undefined) falloutPresence[i] = 1;
      }
    }

    for (const unit of game.units()) {
      const i = cellByTile[unit.tile()];
      if (i === undefined) continue;
      const owner = unit.owner();
      if (me === null || owner.id() !== me.id()) enemyStructCount[i]++;
    }

    const packed = new Uint8Array(RETINA_PLANES.length * cellCount);
    const put = (planeIndex: number, cell: number, value: number) => {
      packed[planeIndex * cellCount + cell] = value;
    };
    for (let i = 0; i < cellCount; i++) {
      const landHere = cache.landTileCount[i];
      put(0, i, ownCount[i] > 0 ? 255 : 0);
      put(
        1,
        i,
        landHere > 0
          ? Math.max(0, Math.min(255, Math.round((enemyLandCount[i] / landHere) * 255)))
          : 0,
      );
      const neutral = landHere - ownCount[i] - enemyLandCount[i];
      put(2, i, neutral > 0 ? 255 : 0);
      put(3, i, cache.waterPresence[i] ? 255 : 0);
      put(4, i, cache.mountainPresence[i] ? 255 : 0);
      put(5, i, falloutPresence[i] ? 255 : 0);
      put(6, i, borderUnderAttack[i] ? 255 : 0);
      put(
        7,
        i,
        Math.max(
          0,
          Math.min(255, Math.round((enemyStructCount[i] / RETINA_STRUCT_NORM) * 255)),
        ),
      );
    }

    return {
      planes: [...RETINA_PLANES],
      size,
      dtype: "uint8",
      data: Buffer.from(
        packed.buffer,
        packed.byteOffset,
        packed.byteLength,
      ).toString("base64"),
    };
  }

  private getRetinaCache(size: number): RetinaCache {
    const game = this.requireRunner().game;
    const numLandTiles = game.numLandTiles();
    if (
      this.retinaCache !== null &&
      this.retinaCache.size === size &&
      this.retinaCache.numLandTiles === numLandTiles
    ) {
      return this.retinaCache;
    }
    const cellCount = size * size;
    const cellByTile: number[] = [];
    const landTileCount = new Array<number>(cellCount).fill(0);
    const landPresence = new Uint8Array(cellCount);
    const waterPresence = new Uint8Array(cellCount);
    const mountainPresence = new Uint8Array(cellCount);
    const landTiles: TileRef[] = [];
    const w = Math.max(1, game.width());
    const h = Math.max(1, game.height());
    game.forEachTile((tile) => {
      const bx = Math.min(size - 1, Math.floor((game.x(tile) / w) * size));
      const by = Math.min(size - 1, Math.floor((game.y(tile) / h) * size));
      const i = by * size + bx;
      cellByTile[tile] = i;
      if (game.isLand(tile)) {
        landPresence[i] = 1;
        landTileCount[i]++;
        landTiles.push(tile);
        if (game.terrainType(tile) === TerrainType.Mountain) mountainPresence[i] = 1;
      } else {
        waterPresence[i] = 1;
      }
    });
    this.retinaCache = {
      size,
      numLandTiles,
      cellByTile,
      landTileCount,
      landTiles,
      landPresence,
      waterPresence,
      mountainPresence,
    };
    return this.retinaCache;
  }

  /**
   * Multi-agent step: apply one action per controlled player in a single tick
   * advance, then return per-player observations/rewards/dones. Actions are
   * indices into each player's last candidate list (same convention as
   * `step`). A dead/finished slot's action is ignored (noop). The game engine
   * already routes each intent to its own player via the stamped clientID.
   */
  stepMulti(actionIndices: number[]): HeadlessMultiStepResult {
    const started = performance.now();
    this.requireRunner();

    // Snapshot before-metrics for every controlled player.
    const before = new Map<ClientID, Metrics>();
    for (const cid of this.controlledClientIDs) {
      before.set(cid, this.previousMetricsByClient.get(cid) ?? this.metrics(cid));
    }

    // Build one stamped intent per (alive, acting) controlled player.
    const stampedIntents: StampedIntent[] = [];
    this.controlledClientIDs.forEach((cid, i) => {
      if (this.isPlayerDone(cid)) return;
      const candidates =
        this.lastCandidatesByClient.get(cid) ?? this.generateCandidates(cid);
      const actionIndex = actionIndices[i] ?? 0;
      const candidate = candidates[actionIndex] ?? candidates[0];
      if (candidate === undefined) return;
      this.actionTranscript.push({ turn: this.turnNumber, candidate });
      if (candidate.kind === "intent" && candidate.intent !== undefined) {
        stampedIntents.push({
          ...candidate.intent,
          clientID: cid,
        } as StampedIntent);
      }
    });

    // Advance the simulation once (intents applied on the first micro-tick).
    const simStarted = performance.now();
    for (let i = 0; i < this.config.decisionInterval; i++) {
      if (this.isDone()) break;
      this.enqueueTurn(i === 0 ? stampedIntents : []);
      this.runner!.executeNextTick();
      if (this.lastError !== null) {
        throw new Error(this.lastError.errMsg);
      }
    }
    const simMs = performance.now() - simStarted;

    // Per-player reward + observation; update previous-metrics cache.
    const observations: HeadlessObservation[] = [];
    const rewards: number[] = [];
    const dones: boolean[] = [];
    for (const cid of this.controlledClientIDs) {
      const after = this.metrics(cid);
      rewards.push(this.reward(before.get(cid)!, after, cid));
      this.previousMetricsByClient.set(cid, after);
      observations.push(this.observe(cid).observation);
      dones.push(this.isPlayerDone(cid));
    }
    // Keep the single-agent hero state coherent for any mixed callers.
    this.previousMetrics = this.previousMetricsByClient.get(this.clientID) ?? null;
    this.decisionNumber++;

    const winner = this.winner();
    return {
      observations,
      rewards,
      dones,
      done: this.isDone(),
      info: {
        winner,
        heroWon: isWinnerForClient(winner, this.clientID),
        winnerControlled:
          winner !== undefined &&
          this.controlledClientIDs.some((cid) =>
            isWinnerForClient(winner, cid),
          ),
        turnNumber: this.turnNumber,
        tick: this.runner?.game.ticks() ?? 0,
        simMs,
        totalMs: performance.now() - started,
        liveOpponents: this.liveOpponentCounts(),
      },
    };
  }

  stepIntents(intents: Intent[]): HeadlessStepResult {
    const started = performance.now();
    this.requireRunner();
    const prepared = this.prepareRawIntents(intents.slice(0, MAX_RAW_INTENTS_PER_STEP));
    const rewardAdjustment = prepared.accepted * 0.01 - prepared.rejected * 0.05;
    return this.advanceWithIntents(
      prepared.stamped,
      started,
      {},
      rewardAdjustment,
      {
        rawIntentAccepted: prepared.accepted,
        rawIntentRejected: prepared.rejected,
        rawIntentReasons: prepared.reasons,
      },
    );
  }

  stepRaw(slots: RawActionSlot[]): HeadlessStepResult {
    const rawStarted = performance.now();
    const built: Intent[] = [];
    const buildReasons: string[] = [];
    let ignored = 0;
    const player = this.requireRunner().game.playerByClientID(this.clientID);
    for (const slot of slots.slice(0, MAX_RAW_INTENTS_PER_STEP)) {
      const result = this.rawSlotToIntent(slot, player);
      if (result.intent !== undefined) {
        built.push(result.intent);
      } else if (result.reason === "noop") {
        ignored += 1;
      } else if (result.reason !== undefined) {
        buildReasons.push(result.reason);
      }
    }
    const result = this.stepIntents(built);
    const priorReasons = Array.isArray(result.info.rawIntentReasons)
      ? result.info.rawIntentReasons
      : [];
    result.info.rawSlotCount = slots.length;
    result.info.rawSlotBuilt = built.length;
    result.info.rawSlotIgnored = ignored;
    result.info.rawSlotRejected = buildReasons.length;
    result.info.rawSlotBuildMs = performance.now() - rawStarted;
    result.info.rawIntentReasons = [...priorReasons, ...buildReasons].slice(0, 16);
    result.reward -= buildReasons.length * 0.05;
    return result;
  }

  private prepareRawIntents(intents: Intent[]): {
    stamped: StampedIntent[];
    accepted: number;
    rejected: number;
    reasons: string[];
  } {
    const stamped: StampedIntent[] = [];
    const reasons: string[] = [];
    const player = this.requireRunner().game.playerByClientID(this.clientID);
    for (const intent of intents) {
      const validation = this.validateRawIntent(intent, player);
      if (validation.reason !== undefined) {
        reasons.push(validation.reason);
        continue;
      }
      stamped.push({ ...validation.intent!, clientID: this.clientID } as StampedIntent);
      this.actionTranscript.push({
        turn: this.turnNumber,
        candidate: {
          kind: "intent",
          label: `raw:${validation.intent!.type}`,
          intent: validation.intent,
          features: [],
        },
      });
    }
    return {
      stamped,
      accepted: stamped.length,
      rejected: reasons.length,
      reasons: reasons.slice(0, 16),
    };
  }

  private rawSlotToIntent(
    slot: RawActionSlot,
    player: Player | null,
  ): RawIntentResult {
    const type = slot.type ?? "noop";
    if (type === "noop") return { reason: "noop" };
    if (player === null) return { reason: `${type}:missing_player` };

    const tile = this.tileFromRawSlot(slot);
    if (tile === null && type !== "attack") return { reason: `${type}:missing_tile` };

    if (type === "spawn") {
      return {
        intent: {
          type: "spawn",
          tile: tile!,
        },
      };
    }
    if (type === "attack") {
      const target = this.targetFromRawSlot(slot, player);
      return {
        intent: {
          type: "attack",
          targetID: target,
          troops: this.troopsFromRatio(player, slot.troopRatio),
        },
      };
    }
    if (type === "boat") {
      return {
        intent: {
          type: "boat",
          dst: tile!,
          troops: this.troopsFromRatio(player, slot.troopRatio),
        },
      };
    }
    if (type === "build_unit") {
      const unit = this.playerBuildableFromRawSlot(slot);
      if (unit === null) return { reason: "build_unit:invalid_unit" };
      return {
        intent: {
          type: "build_unit",
          unit,
          tile: tile!,
          rocketDirectionUp: slot.rocketDirectionUp ?? true,
        },
      };
    }
    return { reason: `${type}:unsupported` };
  }

  private validateRawIntent(
    intent: Intent,
    player: Player | null,
  ): RawIntentResult {
    const game = this.requireRunner().game;
    if (player === null) return { reason: `${intent.type}:missing_player` };
    if (intent.type === "spawn") {
      if (player.hasSpawned()) return { reason: "spawn:already_spawned" };
      if (!game.isValidRef(intent.tile)) return { reason: "spawn:invalid_tile" };
      if (!game.isLand(intent.tile)) return { reason: "spawn:not_land" };
      if (game.hasOwner(intent.tile)) return { reason: "spawn:owned" };
      if (game.hasFallout(intent.tile)) return { reason: "spawn:fallout" };
      return { intent };
    }
    if (!player.hasSpawned() || !player.isAlive()) {
      return { reason: `${intent.type}:not_spawned_or_dead` };
    }
    if (intent.type === "attack") {
      if (intent.targetID === null) {
        if (!hasLandBorderWithTerraNullius(game, player)) {
          return { reason: "attack:terra_not_bordering" };
        }
        return { intent };
      }
      if (!game.hasPlayer(intent.targetID)) return { reason: "attack:missing_target" };
      const target = game.player(intent.targetID);
      if (!target.isAlive()) return { reason: "attack:target_dead" };
      if (!player.canAttackPlayer(target, true)) {
        return { reason: "attack:cannot_attack_target" };
      }
      return { intent };
    }
    if (intent.type === "boat") {
      if (game.config().isUnitDisabled(UnitType.TransportShip)) {
        return { reason: "boat:disabled" };
      }
      if (!game.isValidRef(intent.dst)) return { reason: "boat:invalid_dst" };
      if (!game.isLand(intent.dst)) return { reason: "boat:dst_not_land" };
      if (game.hasOwner(intent.dst) && game.owner(intent.dst) === player) {
        return { reason: "boat:own_dst" };
      }
      if (player.canBuild(UnitType.TransportShip, intent.dst) === false) {
        return { reason: "boat:unreachable_or_limit" };
      }
      return { intent };
    }
    if (intent.type === "build_unit") {
      if (!PlayerBuildable.has(intent.unit)) return { reason: "build:invalid_unit" };
      if (!game.isValidRef(intent.tile)) return { reason: "build:invalid_tile" };
      if (!this.canBuildUnitType(player, intent.unit)) {
        return { reason: `build:${intent.unit}:unaffordable_or_disabled` };
      }
      const spawnTile = player.canBuild(intent.unit, intent.tile);
      if (spawnTile === false) return { reason: `build:${intent.unit}:cannot_build_here` };
      return {
        intent: {
          ...intent,
          tile: spawnTile,
        },
      };
    }
    if (intent.type === "cancel_attack") {
      if (!player.outgoingAttacks().some((attack) => attack.id() === intent.attackID)) {
        return { reason: "cancel_attack:missing_attack" };
      }
      return { intent };
    }
    if (intent.type === "cancel_boat") {
      if (!player.units(UnitType.TransportShip).some((unit) => unit.id() === intent.unitID)) {
        return { reason: "cancel_boat:missing_boat" };
      }
      return { intent };
    }
    if (intent.type === "move_warship") {
      if (!game.isValidRef(intent.tile) || !game.isWater(intent.tile)) {
        return { reason: "move_warship:invalid_water_tile" };
      }
      const ownedWarships = new Set(
        player.units(UnitType.Warship).map((unit) => unit.id()),
      );
      if (!intent.unitIds.some((unitID) => ownedWarships.has(unitID))) {
        return { reason: "move_warship:missing_warship" };
      }
      return { intent };
    }
    return { intent };
  }

  private tileFromRawSlot(slot: RawActionSlot): TileRef | null {
    const game = this.requireRunner().game;
    if (typeof slot.tile === "number" && game.isValidRef(Math.floor(slot.tile))) {
      return Math.floor(slot.tile) as TileRef;
    }
    if (typeof slot.x !== "number" || typeof slot.y !== "number") return null;
    const x = Math.min(
      game.width() - 1,
      Math.max(0, Math.floor(clamp(slot.x, 0, 1) * game.width())),
    );
    const y = Math.min(
      game.height() - 1,
      Math.max(0, Math.floor(clamp(slot.y, 0, 1) * game.height())),
    );
    return game.ref(x, y);
  }

  private targetFromRawSlot(
    slot: RawActionSlot,
    player: Player,
  ): PlayerID | null {
    const game = this.requireRunner().game;
    if (slot.targetID === null) return null;
    if (slot.targetID !== undefined && game.hasPlayer(slot.targetID)) {
      return slot.targetID;
    }
    const targets = game
      .players()
      .filter((other) => other.id() !== player.id() && other.isAlive())
      .sort((a, b) => b.numTilesOwned() - a.numTilesOwned());
    if (targets.length === 0) return null;
    const index = Math.abs(Math.floor(slot.targetIndex ?? 0)) % (targets.length + 1);
    if (index === 0) return null;
    return targets[index - 1].id();
  }

  private troopsFromRatio(player: Player, ratio: number | undefined): number {
    return Math.max(1, Math.floor(player.troops() * clamp(ratio ?? 0.3, 0.01, 1)));
  }

  private playerBuildableFromRawSlot(slot: RawActionSlot): PlayerBuildableUnitType | null {
    if (slot.unit !== undefined && PlayerBuildable.has(slot.unit)) return slot.unit;
    return null;
  }

  private advanceWithIntents(
    stampedIntents: StampedIntent[],
    started: number,
    timings: Record<string, number> = {},
    rewardAdjustment = 0,
    extraInfo: Record<string, unknown> = {},
  ): HeadlessStepResult {
    const metricsStarted = performance.now();
    const before = this.previousMetrics ?? this.metrics();
    const preMetricsMs = performance.now() - metricsStarted;
    const simStarted = performance.now();
    for (let i = 0; i < this.config.decisionInterval; i++) {
      if (this.isDone()) break;
      this.enqueueTurn(i === 0 ? stampedIntents : []);
      this.runner!.executeNextTick();
      if (this.lastError !== null) {
        throw new Error(this.lastError.errMsg);
      }
    }
    const simMs = performance.now() - simStarted;

    const rewardStarted = performance.now();
    const after = this.metrics();
    const reward = this.reward(before, after) + rewardAdjustment;
    this.previousMetrics = after;
    const rewardMs = performance.now() - rewardStarted;
    this.decisionNumber++;
    return this.result(
      reward,
      {
        ...timings,
        preMetricsMs,
        simMs,
        rewardMs,
      },
      started,
      extraInfo,
    );
  }

  transcript(): Array<{ turn: number; candidate: ActionCandidate }> {
    return this.actionTranscript;
  }

  gameRecord(): GameRecord {
    const runner = this.requireRunner();
    if (this.gameStartInfo === null) {
      throw new Error("environment has not been reset");
    }
    const stats = runner.game.stats().stats()[this.clientID];
    const playerRecord: PlayerRecord = {
      clientID: this.clientID,
      username: USERNAME,
      clanTag: null,
      persistentID: null,
      stats,
    };
    const end = this.startedAt + this.turnNumber * 100;
    const partial = createPartialGameRecord(
      this.gameStartInfo.gameID,
      this.gameStartInfo.config,
      [playerRecord],
      this.turns,
      this.startedAt,
      end,
      this.winner(),
      this.gameStartInfo.lobbyCreatedAt,
      this.gameStartInfo.visibleAt,
    );
    return {
      ...partial,
      turns: this.turns,
      gitCommit: "DEV",
      subdomain: "local",
      domain: "localhost",
    };
  }

  private preSpawnOpponents(): number {
    const expected = this.expectedOpponentCounts();
    if (expected.bots + expected.nations === 0) {
      return 0;
    }

    let turns = 0;
    while (!this.opponentsReady(expected)) {
      if (turns >= MAX_OPPONENT_PRESPAWN_TURNS) {
        const live = this.liveOpponentCounts();
        throw new Error(
          `opponents failed to spawn before ML spawn: expected ${formatOpponentCounts(
            expected,
          )}, live ${formatOpponentCounts(live)}`,
        );
      }
      this.enqueueTurn([]);
      this.runner!.executeNextTick();
      if (this.lastError !== null) {
        throw new Error(this.lastError.errMsg);
      }
      turns++;
    }
    return turns;
  }

  private expectedOpponentCounts(): OpponentCounts {
    const game = this.requireRunner().game;
    return {
      bots: this.config.bots,
      nations: this.config.nations === "disabled" ? 0 : game.nations().length,
    };
  }

  private liveOpponentCounts(): OpponentCounts {
    const players = this.requireRunner().game.players();
    return {
      bots: players.filter((player) => player.type() === PlayerType.Bot).length,
      nations: players.filter((player) => player.type() === PlayerType.Nation)
        .length,
    };
  }

  private opponentsReady(expected: OpponentCounts): boolean {
    const live = this.liveOpponentCounts();
    // Tolerate ONE missing nation: some maps (Asia, Mena) have a nation whose
    // designated spawn never fires, which deterministically failed 4/24 eval
    // games. The guard still catches the June failure mode (mass no-spawn /
    // empty-map wins) — 24/25 nations is a real lobby, 0/25 is not.
    const nationFloor = Math.max(0, expected.nations - 1);
    return live.bots >= expected.bots && live.nations >= nationFloor;
  }

  private enqueueTurn(intents: StampedIntent[]): void {
    const turn: Turn = {
      turnNumber: this.turnNumber,
      intents,
    };
    this.runner!.addTurn(turn);
    this.turns.push(turn);
    this.turnNumber++;
  }

  private result(
    reward: number,
    timings: Record<string, number> = {},
    started?: number,
    extraInfo: Record<string, unknown> = {},
  ): HeadlessStepResult {
    const observeStarted = performance.now();
    const { observation, timings: observeTimings } = this.observe();
    const mergedTimings: Record<string, number> = {
      ...timings,
      ...observeTimings,
      observeMs: performance.now() - observeStarted,
    };
    if (started !== undefined) {
      mergedTimings.totalMs = performance.now() - started;
    }
    this.lastTimings = mergedTimings;
    const winner = this.winner();
    return {
      observation,
      reward,
      done: this.isDone(),
      info: {
        actionTranscriptLength: this.actionTranscript.length,
        map: this.config.map,
        difficulty: this.config.difficulty,
        winner,
        won: isWinnerForClient(winner, this.clientID),
        turnNumber: this.turnNumber,
        tick: this.runner?.game.ticks() ?? 0,
        lastUpdateTick: this.lastUpdate?.tick,
        liveOpponents: this.liveOpponentCounts(),
        expectedOpponents: this.expectedOpponentCounts(),
        maxTurnNumber: this.maxTurnNumber,
        timings: this.lastTimings,
        ...extraInfo,
      },
    };
  }

  private observe(cid: ClientID = this.clientID): {
    observation: HeadlessObservation;
    timings: Record<string, number>;
  } {
    const runner = this.requireRunner();
    const game = runner.game;
    const me = game.playerByClientID(cid);
    const candidatesStarted = performance.now();
    const previousCandidateTimings = this.candidateTimings;
    const candidateTimings: Record<string, number> = {};
    this.candidateTimings = candidateTimings;
    let candidates: ActionCandidate[];
    try {
      candidates = this.generateCandidates(cid);
    } finally {
      this.candidateTimings = previousCandidateTimings;
    }
    // Cache for the matching step path: single-agent `step` reads
    // `lastCandidates`; multi-agent `stepMulti` reads `lastCandidatesByClient`.
    this.lastCandidatesByClient.set(cid, candidates);
    if (cid === this.clientID) this.lastCandidates = candidates;
    const observeCandidatesMs = performance.now() - candidatesStarted;
    const metricsStarted = performance.now();
    const metrics = this.metrics(cid);
    const observeMetricsMs = performance.now() - metricsStarted;
    const playersStarted = performance.now();
    const players = game
      .players()
      .slice()
      .sort((a, b) => b.numTilesOwned() - a.numTilesOwned())
      .slice(0, this.config.maxPlayers)
      .map((player) => summarizePlayer(player, me?.id() ?? null));
    const observePlayersMs = performance.now() - playersStarted;
    const spatialStarted = performance.now();
    const spatial = this.spatialPlanes(cid);
    const spatialMs = performance.now() - spatialStarted;

    const observationCandidates = this.config.compactCandidates
      ? candidates.map((candidate) => ({ features: candidate.features }))
      : candidates;
    const spatialBinary = this.config.compactSpatial
      ? encodeSpatialBinary(spatial, this.config.spatialSize)
      : undefined;

    return {
      observation: {
        tick: game.ticks(),
        turnNumber: this.turnNumber,
        spatialSize: this.config.spatialSize,
        spatial: this.config.compactSpatial ? {} : spatial,
        ...(spatialBinary !== undefined ? { spatialBinary } : {}),
        vector: [
          metrics.myShare,
          metrics.myTiles / Math.max(1, game.numLandTiles()),
          metrics.troops / 10_000_000,
          metrics.gold / 10_000_000,
          metrics.rank / Math.max(1, game.players().length),
          metrics.enemiesAlive / Math.max(1, game.players().length - 1),
          game.inSpawnPhase() ? 1 : 0,
          this.isDone() ? 1 : 0,
        ],
        players,
        candidates: observationCandidates,
        candidateCount: candidates.length,
        candidateXYIndex: ACTION_TYPE_NAMES.length,
      },
      timings: {
        observeCandidatesMs,
        ...candidateTimings,
        observeMetricsMs,
        observePlayersMs,
        spatialMs,
      },
    };
  }

  private spatialPlanes(cid: ClientID = this.clientID): Record<string, number[]> {
    const game = this.requireRunner().game;
    const me = game.playerByClientID(cid);
    const size = this.config.spatialSize;
    const cache = this.getSpatialCache(size);
    const cellCount = size * size;
    const denom = cache.denom;
    const planes: Record<string, number[]> = {
      land: cache.staticPlanes.land.slice(),
      own: new Array<number>(cellCount).fill(0),
      enemy: new Array<number>(cellCount).fill(0),
      nation: new Array<number>(cellCount).fill(0),
      unowned: new Array<number>(cellCount).fill(0),
      fallout: new Array<number>(cellCount).fill(0),
      shore: cache.staticPlanes.shore.slice(),
      structure: new Array<number>(cellCount).fill(0),
      ownBorder: new Array<number>(cellCount).fill(0),
      enemyBorder: new Array<number>(cellCount).fill(0),
      ownTroops: new Array<number>(cellCount).fill(0),
      enemyTroops: new Array<number>(cellCount).fill(0),
      ownWarship: new Array<number>(cellCount).fill(0),
      enemyWarship: new Array<number>(cellCount).fill(0),
      ownTransport: new Array<number>(cellCount).fill(0),
      enemyTransport: new Array<number>(cellCount).fill(0),
      port: new Array<number>(cellCount).fill(0),
      city: new Array<number>(cellCount).fill(0),
      missileSilo: new Array<number>(cellCount).fill(0),
      defensePost: new Array<number>(cellCount).fill(0),
      sam: new Array<number>(cellCount).fill(0),
      outgoingAttack: new Array<number>(cellCount).fill(0),
      incomingAttack: new Array<number>(cellCount).fill(0),
    };

    planes.unowned = cache.landCounts.slice();
    for (const owner of game.players()) {
      const isSelf = owner.id() === me?.id();
      const isNation = owner.type() === PlayerType.Nation;
      const troopDensity = clamp(
        owner.troops() / Math.max(1, owner.numTilesOwned()) / 1_000_000,
        0,
        1,
      );
      for (const tile of owner.tiles()) {
        const i = cache.cellByTile[tile];
        if (i === undefined) continue;
        planes.unowned[i]--;
        if (isSelf) {
          planes.own[i]++;
          planes.ownTroops[i] += troopDensity;
        } else {
          planes.enemy[i]++;
          planes.enemyTroops[i] += troopDensity;
          if (isNation) planes.nation[i]++;
        }
      }
      const borderPlane = isSelf ? planes.ownBorder : planes.enemyBorder;
      for (const tile of owner.borderTiles()) {
        const i = cache.cellByTile[tile];
        if (i !== undefined) borderPlane[i]++;
      }
    }

    if (game.numTilesWithFallout() > 0) {
      for (const tile of cache.landTiles) {
        if (!game.hasFallout(tile)) continue;
        const i = cache.cellByTile[tile];
        if (i !== undefined) planes.fallout[i]++;
      }
    }

    for (const unit of game.units()) {
      const i = cache.cellByTile[unit.tile()];
      if (i === undefined) continue;
      const owner = unit.owner();
      planes.structure[i]++;
      if (owner.id() === me?.id()) {
        if (unit.type() === UnitType.Warship) planes.ownWarship[i]++;
        if (unit.type() === UnitType.TransportShip) planes.ownTransport[i]++;
      } else {
        if (unit.type() === UnitType.Warship) planes.enemyWarship[i]++;
        if (unit.type() === UnitType.TransportShip) planes.enemyTransport[i]++;
      }
      if (unit.type() === UnitType.Port) planes.port[i]++;
      if (unit.type() === UnitType.City) planes.city[i]++;
      if (unit.type() === UnitType.MissileSilo) planes.missileSilo[i]++;
      if (unit.type() === UnitType.DefensePost) planes.defensePost[i]++;
      if (unit.type() === UnitType.SAMLauncher) planes.sam[i]++;
    }

    for (const player of game.players()) {
      for (const attack of player.outgoingAttacks()) {
        const plane =
          player.id() === me?.id() ? planes.outgoingAttack : planes.incomingAttack;
        for (const tile of attack.clusteredPositions()) {
          const i = cache.cellByTile[tile];
          if (i !== undefined) {
            plane[i] += clamp(attack.troops() / 1_000_000, 0, 1);
          }
        }
      }
    }

    for (const [name, values] of Object.entries(planes)) {
      if (name === "land" || name === "shore") continue;
      for (let i = 0; i < values.length; i++) {
        values[i] = denom[i] > 0 ? values[i] / denom[i] : 0;
      }
    }
    return planes;
  }

  private getSpatialCache(size: number): SpatialCache {
    const game = this.requireRunner().game;
    const numLandTiles = game.numLandTiles();
    if (
      this.spatialCache !== null &&
      this.spatialCache.size === size &&
      this.spatialCache.numLandTiles === numLandTiles
    ) {
      return this.spatialCache;
    }
    const cellCount = size * size;
    const cellByTile: number[] = [];
    const denom = new Array<number>(cellCount).fill(0);
    const landCounts = new Array<number>(cellCount).fill(0);
    const land = new Array<number>(cellCount).fill(0);
    const shore = new Array<number>(cellCount).fill(0);
    const landTiles: TileRef[] = [];
    game.forEachTile((tile) => {
      const bx = Math.min(
        size - 1,
        Math.floor((game.x(tile) / Math.max(1, game.width())) * size),
      );
      const by = Math.min(
        size - 1,
        Math.floor((game.y(tile) / Math.max(1, game.height())) * size),
      );
      const i = by * size + bx;
      cellByTile[tile] = i;
      denom[i]++;
      if (game.isLand(tile)) {
        land[i]++;
        landCounts[i]++;
        landTiles.push(tile);
      }
      if (game.isShore(tile)) shore[i]++;
    });
    for (let i = 0; i < cellCount; i++) {
      land[i] = denom[i] > 0 ? land[i] / denom[i] : 0;
      shore[i] = denom[i] > 0 ? shore[i] / denom[i] : 0;
    }
    this.spatialCache = {
      size,
      numLandTiles,
      cellByTile,
      denom,
      landCounts,
      landTiles,
      staticPlanes: { land, shore },
    };
    return this.spatialCache;
  }

  private generateCandidates(cid: ClientID = this.clientID): ActionCandidate[] {
    const runner = this.requireRunner();
    const game = runner.game;
    const player = game.playerByClientID(cid);
    const candidates: ActionCandidate[] = [];
    const previousFeatureContext = this.candidateFeatureContext;
    this.candidateFeatureContext = createCandidateFeatureContext(
      game,
      cid,
    );
    const add = ((candidate: ActionCandidate) => {
      if (candidates.length < this.config.maxActions) candidates.push(candidate);
    }) as CandidateAdder;
    add.hasCapacity = () => candidates.length < this.config.maxActions;
    try {
      add({
        kind: "noop",
        label: "noop",
        features: candidateFeatures(
          game,
          "noop",
          undefined,
          this.config.localFeatureRadii,
          this.candidateFeatureContext,
        ),
      });

      if (player === null) return candidates;

      if (!player.hasSpawned()) {
        this.timeCandidateSection("spawnCandidatesMs", () => {
          const maxSpawns = this.config.actionProfile === "bot-lite" ? 32 : 64;
          for (const tile of sampleEvenly(this.unownedLandTiles(), maxSpawns)) {
            add(this.intentCandidate(game, "spawn", { type: "spawn", tile }));
          }
        });
        return candidates;
      }

      this.timeCandidateSection("attackCandidatesMs", () =>
        this.addAttackCandidates(add, player),
      );
      if (this.shouldGenerateCandidateGroup(this.config.boatCandidateInterval)) {
        this.timeCandidateSection("boatCandidatesMs", () =>
          this.addBoatCandidates(add, player),
        );
      } else {
        this.addCandidateTiming("boatCandidatesSkipped", 1);
      }
      if (this.shouldGenerateCandidateGroup(this.config.buildCandidateInterval)) {
        this.timeCandidateSection("buildCandidatesMs", () =>
          this.addBuildCandidates(add, player),
        );
      } else {
        this.addCandidateTiming("buildCandidatesSkipped", 1);
      }
      if (this.config.actionProfile !== "bot-lite") {
        this.timeCandidateSection("upgradeDeleteCandidatesMs", () =>
          this.addUpgradeAndDeleteCandidates(add, player),
        );
      }
      if (this.config.actionProfile === "full") {
        this.timeCandidateSection("diplomacyCandidatesMs", () =>
          this.addDiplomacyCandidates(add, player),
        );
        this.timeCandidateSection("communicationCandidatesMs", () =>
          this.addCommunicationCandidates(add, player),
        );
      }
      if (this.config.actionProfile !== "bot-lite") {
        this.timeCandidateSection("warshipCandidatesMs", () =>
          this.addWarshipCandidates(add, player),
        );
      }
      return candidates;
    } finally {
      this.candidateFeatureContext = previousFeatureContext;
    }
  }

  private intentCandidate(
    game: GameRunner["game"],
    label: string,
    intent: Intent,
  ): ActionCandidate {
    return intentCandidate(
      game,
      label,
      intent,
      this.config.localFeatureRadii,
      this.candidateFeatureContext,
    );
  }

  private timeCandidateSection<T>(key: string, fn: () => T): T {
    if (this.candidateTimings === null) return fn();
    const started = performance.now();
    try {
      return fn();
    } finally {
      this.addCandidateTiming(key, performance.now() - started);
    }
  }

  private addCandidateTiming(key: string, elapsedMs: number): void {
    if (this.candidateTimings === null) return;
    this.candidateTimings[key] =
      (this.candidateTimings[key] ?? 0) + elapsedMs;
  }

  private shouldGenerateCandidateGroup(interval: number): boolean {
    const cleanInterval = Math.max(1, Math.floor(interval || 1));
    return this.decisionNumber % cleanInterval === 0;
  }

  private addAttackCandidates(
    add: CandidateAdder,
    player: Player,
  ): void {
    const game = this.requireRunner().game;
    if (hasLandBorderWithTerraNullius(game, player)) {
      for (const ratio of TROOP_RATIOS) {
        if (!add.hasCapacity()) return;
        add(
          this.intentCandidate(game, "attack:terra", {
            type: "attack",
            targetID: null,
            troops: Math.max(1, Math.floor(player.troops() * ratio)),
          }),
        );
      }
    }

    const targets = game
      .players()
      .filter((p) => p.id() !== player.id() && p.isAlive())
      .sort((a, b) => {
        const borderDelta =
          Number(player.sharesBorderWith(b)) - Number(player.sharesBorderWith(a));
        return borderDelta || b.numTilesOwned() - a.numTilesOwned();
      })
      .slice(0, this.config.actionProfile === "bot-lite" ? 6 : 12);
    for (const target of targets) {
      if (!player.canAttackPlayer(target, true)) continue;
      for (const ratio of TROOP_RATIOS) {
        if (!add.hasCapacity()) return;
        add(
          this.intentCandidate(game, `attack:${target.type()}`, {
            type: "attack",
            targetID: target.id(),
            troops: Math.max(1, Math.floor(player.troops() * ratio)),
          }),
        );
      }
    }

    const maxCancels = this.config.actionProfile === "bot-lite" ? 8 : 24;
    for (const attack of player.outgoingAttacks().slice(0, maxCancels)) {
      if (!add.hasCapacity()) return;
      add(
        this.intentCandidate(game, "cancel_attack", {
          type: "cancel_attack",
          attackID: attack.id(),
        }),
      );
    }
  }

  private addBoatCandidates(
    add: CandidateAdder,
    player: Player,
  ): void {
    const game = this.requireRunner().game;
    if (game.config().isUnitDisabled(UnitType.TransportShip)) return;
    const candidateTiles = this.shoreLandTiles().filter(
      (tile) =>
        game.isLand(tile) && (!game.hasOwner(tile) || game.owner(tile) !== player),
    );
    const probeLimit = this.config.boatDestinationProbeLimit;
    const botLite = this.config.actionProfile === "bot-lite";
    const destinationLimit = botLite ? 4 : 16;
    const reachableWaterComponents = this.playerReachableWaterComponents(player);
    const canSpawnTransport = (tile: TileRef): boolean => {
      const component = game.getWaterComponent(tile);
      return component !== null && reachableWaterComponents.has(component);
    };
    const destinations =
      probeLimit === "full"
        ? sampleEvenly(
            candidateTiles.filter(
              (tile) => canSpawnTransport(tile),
            ),
            destinationLimit,
          )
        : sampleEvenly(candidateTiles, probeLimit)
            .filter((tile) => canSpawnTransport(tile))
            .slice(0, destinationLimit);
    for (const dst of destinations) {
      const ratios = botLite ? [0.25, 0.5] : [0.15, 0.3, 0.5];
      for (const ratio of ratios) {
        if (!add.hasCapacity()) return;
        add(
          this.intentCandidate(game, "boat", {
            type: "boat",
            dst,
            troops: Math.max(1, Math.floor(player.troops() * ratio)),
          }),
        );
      }
    }

    const maxBoatCancels = this.config.actionProfile === "bot-lite" ? 4 : 12;
    for (const unit of player.units(UnitType.TransportShip).slice(0, maxBoatCancels)) {
      if (!add.hasCapacity()) return;
      add(
        this.intentCandidate(game, "cancel_boat", {
          type: "cancel_boat",
          unitID: unit.id(),
        }),
      );
    }
  }

  private playerReachableWaterComponents(player: Player): Set<number> {
    const game = this.requireRunner().game;
    const components = new Set<number>();
    for (const tile of player.borderTiles()) {
      if (!game.isLand(tile) || !game.isShore(tile)) continue;
      const component = game.getWaterComponent(tile);
      if (component !== null) components.add(component);
    }
    return components;
  }

  private addBuildCandidates(
    add: CandidateAdder,
    player: Player,
  ): void {
    const game = this.requireRunner().game;
    const botLite = this.config.actionProfile === "bot-lite";
    const defaultLimit = botLite ? 24 : 80;
    const probeLimit = botLite ? defaultLimit : this.config.buildProbeLimit;
    const probeTiles = this.timeCandidateSection("buildProbeTilesMs", () =>
      this.buildProbeTiles(player),
    );
    const tiles =
      probeLimit === "full"
        ? probeTiles
        : sampleEvenly(probeTiles, probeLimit || defaultLimit);
    const seen = new Set<string>();
    const affordable = new Set(
      PlayerBuildable.types.filter((unitType) =>
        this.canBuildUnitType(player, unitType),
      ),
    );
    const landStructureUnits = LAND_STRUCTURE_BUILD_UNITS.filter(
      (unitType) =>
        affordable.has(unitType) &&
        (!botLite || BOT_LITE_BUILD_UNITS.has(unitType)),
    );
    const specialUnits = SPECIAL_BUILD_UNITS.filter(
      (unitType) =>
        affordable.has(unitType) &&
        (!botLite || BOT_LITE_BUILD_UNITS.has(unitType)),
    );
    for (const tile of tiles) {
      if (!add.hasCapacity()) break;
      let validStructureTiles: TileRef[] | null = null;
      if (landStructureUnits.length > 0) {
        validStructureTiles = this.timeCandidateSection(
          "buildValidStructureTilesMs",
          () => this.validStructureSpawnTiles(player, tile),
        );
        const structureTile = validStructureTiles[0];
        if (structureTile !== undefined) {
          for (const unitType of landStructureUnits) {
            if (!add.hasCapacity()) break;
            this.addBuildIntentCandidate(add, game, seen, unitType, structureTile);
          }
        }
      }
      for (const unitType of specialUnits) {
        if (!add.hasCapacity()) break;
        const started = performance.now();
        const canBuild = player.canBuild(
          unitType,
          tile,
          unitType === UnitType.Port ? validStructureTiles : null,
        );
        this.addCandidateTiming(
          "buildCanBuildMs",
          performance.now() - started,
        );
        if (canBuild === false) continue;
        this.addBuildIntentCandidate(add, game, seen, unitType, canBuild);
      }
    }
  }

  private addBuildIntentCandidate(
    add: CandidateAdder,
    game: GameRunner["game"],
    seen: Set<string>,
    unitType: PlayerBuildableUnitType,
    tile: TileRef,
  ): void {
    if (!add.hasCapacity()) return;
    const key = `${unitType}:${tile}`;
    if (seen.has(key)) return;
    seen.add(key);
    const featureStarted = performance.now();
    add(
      this.intentCandidate(game, `build:${unitType}`, {
        type: "build_unit",
        unit: unitType,
        tile,
        rocketDirectionUp: true,
      }),
    );
    this.addCandidateTiming(
      "buildFeatureMs",
      performance.now() - featureStarted,
    );
  }

  private canBuildUnitType(
    player: Player,
    unitType: PlayerBuildableUnitType,
  ): boolean {
    const game = this.requireRunner().game;
    if (game.config().isUnitDisabled(unitType)) return false;
    const cost = game.config().unitInfo(unitType).cost(game, player);
    if (player.gold() < cost) return false;
    return player.isAlive();
  }

  private validStructureSpawnTiles(player: Player, tile: TileRef): TileRef[] {
    return (
      player as unknown as {
        validStructureSpawnTiles(tile: TileRef): TileRef[];
      }
    ).validStructureSpawnTiles(tile);
  }

  private addUpgradeAndDeleteCandidates(
    add: CandidateAdder,
    player: Player,
  ): void {
    const game = this.requireRunner().game;
    for (const unit of player.units().slice(0, 48)) {
      if (player.canUpgradeUnit(unit)) {
        if (!add.hasCapacity()) return;
        add(
          this.intentCandidate(game, `upgrade:${unit.type()}`, {
            type: "upgrade_structure",
            unit: unit.type(),
            unitId: unit.id(),
          }),
        );
      }
      if (player.canDeleteUnit()) {
        if (!add.hasCapacity()) return;
        add(
          this.intentCandidate(game, `delete:${unit.type()}`, {
            type: "delete_unit",
            unitId: unit.id(),
          }),
        );
      }
    }
  }

  private addDiplomacyCandidates(
    add: CandidateAdder,
    player: Player,
  ): void {
    const game = this.requireRunner().game;
    const others = game
      .players()
      .filter((p) => p.id() !== player.id() && p.isAlive())
      .slice(0, 24);
    for (const other of others) {
      if (player.canSendAllianceRequest(other)) {
        if (!add.hasCapacity()) return;
        add(
          this.intentCandidate(game, "allianceRequest", {
            type: "allianceRequest",
            recipient: other.id(),
          }),
        );
      }
      const alliance = player.allianceWith(other);
      if (alliance !== null) {
        if (alliance.agreedToExtend(player) === false) {
          if (!add.hasCapacity()) return;
          add(
            this.intentCandidate(game, "allianceExtension", {
              type: "allianceExtension",
              recipient: other.id(),
            }),
          );
        }
        if (!add.hasCapacity()) return;
        add(
          this.intentCandidate(game, "breakAlliance", {
            type: "breakAlliance",
            recipient: other.id(),
          }),
        );
      }
      if (player.canTarget(other)) {
        if (!add.hasCapacity()) return;
        add(
          this.intentCandidate(game, "targetPlayer", {
            type: "targetPlayer",
            target: other.id(),
          }),
        );
      }
      if (player.canDonateGold(other)) {
        if (!add.hasCapacity()) return;
        add(
          this.intentCandidate(game, "donate_gold:auto", {
            type: "donate_gold",
            recipient: other.id(),
            gold: null,
          }),
        );
      }
      if (player.canDonateTroops(other)) {
        if (!add.hasCapacity()) return;
        add(
          this.intentCandidate(game, "donate_troops:auto", {
            type: "donate_troops",
            recipient: other.id(),
            troops: null,
          }),
        );
      }
      if (!player.hasEmbargoAgainst(other)) {
        if (!add.hasCapacity()) return;
        add(
          this.intentCandidate(game, "embargo:start", {
            type: "embargo",
            targetID: other.id(),
            action: "start",
          }),
        );
      } else {
        if (!add.hasCapacity()) return;
        add(
          this.intentCandidate(game, "embargo:stop", {
            type: "embargo",
            targetID: other.id(),
            action: "stop",
          }),
        );
      }
    }

    for (const request of player.incomingAllianceRequests().slice(0, 12)) {
      if (!add.hasCapacity()) return;
      add(
        this.intentCandidate(game, "allianceReject", {
          type: "allianceReject",
          requestor: request.requestor().id(),
        }),
      );
    }
    if (player.canEmbargoAll()) {
      if (!add.hasCapacity()) return;
      add(
        this.intentCandidate(game, "embargo_all:start", {
          type: "embargo_all",
          action: "start",
        }),
      );
      if (!add.hasCapacity()) return;
      add(
        this.intentCandidate(game, "embargo_all:stop", {
          type: "embargo_all",
          action: "stop",
        }),
      );
    }
  }

  private addCommunicationCandidates(
    add: CandidateAdder,
    player: Player,
  ): void {
    const game = this.requireRunner().game;
    if (player.canSendEmoji(AllPlayers)) {
      for (const emoji of EMOJIS) {
        if (!add.hasCapacity()) return;
        add(
          this.intentCandidate(game, "emoji:all", {
            type: "emoji",
            recipient: AllPlayers,
            emoji,
          }),
        );
      }
    }
    const others = game
      .players()
      .filter((p) => p.id() !== player.id() && p.isAlive())
      .slice(0, 8);
    for (const other of others) {
      if (player.canSendEmoji(other)) {
        if (!add.hasCapacity()) return;
        add(
          this.intentCandidate(game, "emoji:player", {
            type: "emoji",
            recipient: other.id(),
            emoji: EMOJIS[0],
          }),
        );
      }
      for (const entry of QUICK_CHAT_KEYS) {
        if (entry.requiresPlayer) {
          if (!add.hasCapacity()) return;
          add(
            this.intentCandidate(game, "quick_chat:target", {
              type: "quick_chat",
              recipient: other.id(),
              quickChatKey: entry.key,
              target: other.id(),
            }),
          );
        } else {
          if (!add.hasCapacity()) return;
          add(
            this.intentCandidate(game, "quick_chat", {
              type: "quick_chat",
              recipient: other.id(),
              quickChatKey: entry.key,
            }),
          );
        }
      }
    }
  }

  private addWarshipCandidates(
    add: CandidateAdder,
    player: Player,
  ): void {
    const game = this.requireRunner().game;
    const warships = player.units(UnitType.Warship).slice(0, 8);
    if (warships.length === 0) return;
    const targets = sampleEvenly(
      this.interestingLandTiles().flatMap((tile) => game.neighbors(tile)),
      12,
    ).filter((tile) => game.isWater(tile));
    for (const unit of warships) {
      for (const tile of targets) {
        if (!add.hasCapacity()) return;
        add(
          this.intentCandidate(game, "move_warship", {
            type: "move_warship",
            unitIds: [unit.id()],
            tile,
          }),
        );
      }
    }
  }

  private reward(
    before: Metrics,
    after: Metrics,
    cid: ClientID = this.clientID,
  ): number {
    let reward = -0.002;
    reward += (after.myShare - before.myShare) * 120;
    if (before.myTiles > 0) {
      reward += Math.max(0, before.enemiesAlive - after.enemiesAlive) * 0.75;
      reward += (before.rank - after.rank) * 0.03;
    }
    reward += clamp((after.troops - before.troops) / 1_000_000, -1, 1) * 0.01;
    reward += clamp((after.gold - before.gold) / 1_000_000, -1, 1) * 0.005;
    // Economy credit at the moment it happens: a city's strategic payoff
    // lands thousands of turns after the build action, far beyond what
    // outcome-level credit assignment can bridge. Net count deltas so
    // build/lose churn is not farmable, plus a small holding income so
    // keeping economy alive keeps paying.
    reward += Math.max(0, after.cities - before.cities) * 0.5;
    reward += Math.max(0, after.ports - before.ports) * 0.25;
    reward += 0.002 * Math.min(after.cities, 12);
    // Tech nudges (modest, < the city signal): reward building a SAM/silo and
    // launching a nuke when it happens, since the strategic payoff lands far
    // later than outcome credit can bridge. Net deltas so churn isn't farmable.
    reward += Math.max(0, after.sams - before.sams) * 0.25;
    reward += Math.max(0, after.missileSilos - before.missileSilos) * 0.25;
    reward += Math.min(2, Math.max(0, after.nukesInFlight - before.nukesInFlight)) * 0.4;
    if (!before.alive && after.alive) reward += 0.1;
    if (before.alive && !after.alive) reward -= 8;

    const winner = this.winner();
    if (winner !== undefined) {
      reward += isWinnerForClient(winner, cid) ? 50 : -20;
    } else if (this.turnNumber >= this.maxTurnNumber) {
      reward += after.rank === 1 ? 5 : -5;
    }
    return reward;
  }

  private metrics(cid: ClientID = this.clientID): Metrics {
    const game = this.requireRunner().game;
    const me = game.playerByClientID(cid);
    const players = game.players().filter((p) => p.isAlive());
    const sorted = players
      .slice()
      .sort((a, b) => b.numTilesOwned() - a.numTilesOwned());
    const rank = me === null ? sorted.length : sorted.indexOf(me) + 1;
    return {
      alive: me?.isAlive() ?? false,
      cities: me?.unitCount(UnitType.City) ?? 0,
      enemiesAlive: players.filter((p) => p.id() !== me?.id()).length,
      gold: Number(me?.gold() ?? 0n),
      myShare: (me?.numTilesOwned() ?? 0) / Math.max(1, game.numLandTiles()),
      myTiles: me?.numTilesOwned() ?? 0,
      ports: me?.unitCount(UnitType.Port) ?? 0,
      rank: rank <= 0 ? sorted.length : rank,
      troops: me?.troops() ?? 0,
      missileSilos: me?.unitCount(UnitType.MissileSilo) ?? 0,
      sams: me?.unitCount(UnitType.SAMLauncher) ?? 0,
      nukesInFlight:
        (me?.unitCount(UnitType.AtomBomb) ?? 0) +
        (me?.unitCount(UnitType.HydrogenBomb) ?? 0) +
        (me?.unitCount(UnitType.MIRV) ?? 0),
    };
  }

  private winner(): Winner {
    const winner = this.requireRunner().game.getWinner();
    if (winner === null) return undefined;
    if (typeof winner === "string") return ["team", winner];
    const clientID = winner.clientID();
    return clientID === null ? ["nation", winner.name()] : ["player", clientID];
  }

  private isDone(): boolean {
    const runner = this.runner;
    if (runner === null) return true;
    if (
      runner.game.getWinner() !== null ||
      this.turnNumber >= this.maxTurnNumber
    ) {
      return true;
    }
    // Episode ends once every controlled player has spawned and died. A slot
    // that hasn't spawned yet is not "dead", so this matches the single-agent
    // behavior exactly when there is one controlled player.
    return this.controlledClientIDs.every((cid) => {
      const player = runner.game.playerByClientID(cid);
      return player !== null && player.hasSpawned() && !player.isAlive();
    });
  }

  private isPlayerDone(cid: ClientID): boolean {
    const runner = this.runner;
    if (runner === null) return true;
    if (
      runner.game.getWinner() !== null ||
      this.turnNumber >= this.maxTurnNumber
    ) {
      return true;
    }
    const player = runner.game.playerByClientID(cid);
    return player !== null && player.hasSpawned() && !player.isAlive();
  }

  private unownedLandTiles(): TileRef[] {
    const game = this.requireRunner().game;
    const tiles: TileRef[] = [];
    game.forEachTile((tile) => {
      if (game.isLand(tile) && !game.hasOwner(tile) && !game.hasFallout(tile)) {
        tiles.push(tile);
      }
    });
    return tiles;
  }

  private interestingLandTiles(): TileRef[] {
    const game = this.requireRunner().game;
    const tiles: TileRef[] = [];
    game.forEachTile((tile) => {
      if (!game.isLand(tile)) return;
      if (!game.isShore(tile) && game.hasOwner(tile)) return;
      tiles.push(tile);
    });
    return tiles;
  }

  private shoreLandTiles(): TileRef[] {
    if (this.shoreLandTilesCache !== null) return this.shoreLandTilesCache;
    const game = this.requireRunner().game;
    const tiles: TileRef[] = [];
    game.forEachTile((tile) => {
      if (game.isLand(tile) && game.isShore(tile)) tiles.push(tile);
    });
    this.shoreLandTilesCache = tiles;
    return tiles;
  }

  private buildProbeTiles(player: Player): TileRef[] {
    const game = this.requireRunner().game;
    const tiles = new Set<TileRef>();
    for (const tile of player.tiles()) tiles.add(tile);
    for (const tile of player.borderTiles()) {
      tiles.add(tile);
      for (const neighbor of game.neighbors(tile)) tiles.add(neighbor);
    }
    for (const unit of player.units()) tiles.add(unit.tile());
    return [...tiles].filter((tile) => game.isValidRef(tile));
  }

  private requireRunner(): GameRunner {
    if (this.runner === null) {
      throw new Error("environment has not been reset");
    }
    return this.runner;
  }
}

export function buildGameStartInfo(
  config: Pick<
    HeadlessEpisodeConfig,
    "bots" | "difficulty" | "map" | "nations" | "seed"
  >,
  clientIDs: ClientID | ClientID[] = CLIENT_ID,
): GameStartInfo {
  const ids = Array.isArray(clientIDs) ? clientIDs : [clientIDs];
  return {
    gameID: deterministicID(config.seed, "G"),
    lobbyCreatedAt: 0,
    players: ids.map((clientID, i) => ({
      clientID,
      username: ids.length > 1 ? `${USERNAME}${i + 1}` : USERNAME,
      clanTag: null,
    })),
    config: {
      gameMap: config.map,
      gameMapSize: GameMapSize.Normal,
      gameType: GameType.Singleplayer,
      gameMode: GameMode.FFA,
      difficulty: config.difficulty,
      bots: config.bots,
      nations: config.nations,
      infiniteGold: false,
      infiniteTroops: false,
      instantBuild: false,
      randomSpawn: false,
      donateGold: false,
      donateTroops: false,
      disabledUnits: [],
    } satisfies GameConfig,
  };
}

async function createIsolatedGameRunner(
  gameStart: GameStartInfo,
  clientID: ClientID | undefined,
  mapsRoot: string,
  callBack: (gu: GameUpdateViewData | ErrorUpdate) => void,
  winPercent?: number,
): Promise<GameRunner> {
  const config = new Config(gameStart.config, null, false);
  if (winPercent !== undefined) {
    (config as unknown as { percentageTilesOwnedToWin: () => number }).percentageTilesOwnedToWin =
      () => winPercent;
  }
  const mapLoader = new FileMapLoader(mapsRoot);
  const mapFiles = mapLoader.getMapData(gameStart.config.gameMap);
  const manifest = structuredClone(await mapFiles.manifest());

  const gameMap =
    gameStart.config.gameMapSize === GameMapSize.Normal
      ? await genTerrainFromBin(manifest.map, await mapFiles.mapBin())
      : await genTerrainFromBin(manifest.map4x, await mapFiles.map4xBin());
  const miniGameMap =
    gameStart.config.gameMapSize === GameMapSize.Normal
      ? await genTerrainFromBin(manifest.map4x, await mapFiles.map4xBin())
      : await genTerrainFromBin(manifest.map16x, await mapFiles.map16xBin());

  if (gameStart.config.gameMapSize === GameMapSize.Compact) {
    manifest.nations.forEach((nation) => {
      if (nation.coordinates !== undefined) {
        nation.coordinates = [
          Math.floor(nation.coordinates[0] / 2),
          Math.floor(nation.coordinates[1] / 2),
        ];
      }
    });
    manifest.additionalNations?.forEach((nation) => {
      if (nation.coordinates !== undefined) {
        nation.coordinates = [
          Math.floor(nation.coordinates[0] / 2),
          Math.floor(nation.coordinates[1] / 2),
        ];
      }
    });
  }

  let teamGameSpawnAreas = manifest.teamGameSpawnAreas;
  if (
    gameStart.config.gameMapSize === GameMapSize.Compact &&
    teamGameSpawnAreas !== undefined
  ) {
    const scaled: TeamGameSpawnAreas = {};
    for (const [key, areas] of Object.entries(teamGameSpawnAreas)) {
      scaled[key] = areas.map((area) => ({
        x: Math.floor(area.x / 2),
        y: Math.floor(area.y / 2),
        width: Math.max(1, Math.floor(area.width / 2)),
        height: Math.max(1, Math.floor(area.height / 2)),
      }));
    }
    teamGameSpawnAreas = scaled;
  }

  const random = new PseudoRandom(simpleHash(gameStart.gameID));
  const humans = gameStart.players.map(
    (player) =>
      new PlayerInfo(
        player.username,
        PlayerType.Human,
        player.clientID,
        random.nextID(),
        player.isLobbyCreator ?? false,
        player.clanTag,
      ),
  );
  const nations = createNationsForGame(
    gameStart,
    manifest.nations,
    manifest.additionalNations ?? [],
    humans.length,
    random,
  );
  const game = createGame(
    humans,
    nations,
    gameMap,
    miniGameMap,
    config,
    teamGameSpawnAreas,
  );
  const runner = new GameRunner(
    game,
    new Executor(game, gameStart.gameID, clientID),
    callBack,
  );
  runner.init();
  return runner;
}

function encodeSpatialBinary(
  spatial: Record<string, number[]>,
  size: number,
): HeadlessObservation["spatialBinary"] {
  const order = Object.keys(spatial).sort();
  const cellCount = size * size;
  // Planes are normalized to [0, 1]; uint8 quantization keeps plenty of
  // precision and cuts observation IPC bandwidth 4x vs float32.
  const packed = new Uint8Array(order.length * cellCount);
  for (let planeIndex = 0; planeIndex < order.length; planeIndex++) {
    const values = spatial[order[planeIndex]] ?? [];
    const offset = planeIndex * cellCount;
    for (let i = 0; i < cellCount && i < values.length; i++) {
      packed[offset + i] = Math.max(0, Math.min(255, Math.round(values[i] * 255)));
    }
  }
  return {
    order,
    dtype: "uint8",
    size,
    data: Buffer.from(
      packed.buffer,
      packed.byteOffset,
      packed.byteLength,
    ).toString("base64"),
  };
}

function deterministicID(seed: string, prefix: string): string {
  let hash = 2166136261;
  for (const char of `${prefix}:${seed}`) {
    hash ^= char.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  const alphabet = "123456789abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ";
  let value = hash >>> 0;
  let id = prefix;
  while (id.length < 8) {
    id += alphabet[value % alphabet.length];
    value = Math.floor(value / alphabet.length) ^ 0x9e3779b9;
  }
  return id.slice(0, 8);
}

function summarizePlayer(
  player: Player,
  selfID: PlayerID | null,
): PlayerSummary {
  const units: Partial<Record<UnitType, number>> = {};
  for (const unit of player.units()) {
    units[unit.type()] = (units[unit.type()] ?? 0) + 1;
  }
  return {
    id: player.id(),
    smallID: player.smallID(),
    type: player.type(),
    isSelf: player.id() === selfID,
    isAlive: player.isAlive(),
    tiles: player.numTilesOwned(),
    troops: player.troops(),
    gold: Number(player.gold()),
    outgoingAttacks: player.outgoingAttacks().length,
    incomingAttacks: player.incomingAttacks().length,
    units,
  };
}

function intentCandidate(
  game: GameRunner["game"],
  label: string,
  intent: Intent,
  localFeatureRadii: number[] = [],
  context?: CandidateFeatureContext | null,
): ActionCandidate {
  return {
    kind: "intent",
    label,
    intent,
    features: candidateFeatures(game, label, intent, localFeatureRadii, context),
  };
}

function candidateFeatures(
  game: GameRunner["game"],
  label: string,
  intent?: Intent,
  localFeatureRadii: number[] = [],
  context?: CandidateFeatureContext | null,
): number[] {
  const actor = context?.actor ?? game.playerByClientID(CLIENT_ID);
  const target = intent !== undefined ? targetPlayerForIntent(game, intent) : null;
  const type = intent?.type ?? "noop";
  const oneHot = ACTION_TYPE_NAMES.map((name) => (name === type ? 1 : 0));
  const tile =
    intent !== undefined && "tile" in intent && typeof intent.tile === "number"
      ? intent.tile
      : intent !== undefined && "dst" in intent && typeof intent.dst === "number"
        ? intent.dst
        : undefined;
  const x = tile !== undefined ? game.x(tile) / Math.max(1, game.width()) : 0;
  const y = tile !== undefined ? game.y(tile) / Math.max(1, game.height()) : 0;
  const troops =
    intent !== undefined && "troops" in intent && typeof intent.troops === "number"
      ? intent.troops / 10_000_000
      : 0;
  const labelHash =
    [...label].reduce((sum, char) => sum + char.charCodeAt(0), 0) % 997;
  const unitType =
    intent !== undefined && "unit" in intent && typeof intent.unit === "string"
      ? intent.unit
      : undefined;
  const unitOneHot = ACTION_UNIT_NAMES.map((name) =>
    name === unitType ? 1 : 0,
  );
  return [
    ...oneHot,
    x,
    y,
    troops,
    labelHash / 997,
    ...unitOneHot,
    ...(context?.actorFeatures ?? actorFeatures(game, actor)),
    ...targetFeatures(game, actor, target, context),
    ...tileFeatures(game, actor, tile, context),
    ...intentFeatures(game, actor, target, tile, intent, context),
    ...localPatchFeatures(game, actor, tile, localFeatureRadii, context),
  ];
}

function createCandidateFeatureContext(
  game: GameRunner["game"],
  clientID: ClientID,
): CandidateFeatureContext {
  const actor = game.playerByClientID(clientID);
  const aliveByTiles = game
    .players()
    .filter((player) => player.isAlive())
    .sort((a, b) => b.numTilesOwned() - a.numTilesOwned());
  const rankByID = new Map<PlayerID, number>();
  aliveByTiles.forEach((player, index) => {
    rankByID.set(player.id(), index + 1);
  });
  const context: CandidateFeatureContext = {
    actor,
    actorFeatures: [],
    leaderID: aliveByTiles[0]?.id() ?? null,
    playerCount: game.players().length,
    rankByID,
    targetFeatureCache: new Map(),
    tileFeatureCache: new Map(),
    localPatchFeatureCache: new Map(),
    nearestUnitCache: new Map(),
    nearbyUnitCountCache: new Map(),
    borderContactCache: new Map(),
  };
  context.actorFeatures = actorFeatures(game, actor, context);
  return context;
}

function actorFeatures(
  game: GameRunner["game"],
  actor: Player | null,
  context?: CandidateFeatureContext | null,
): number[] {
  if (actor === null) return new Array(12).fill(0);
  const playerCount = context?.playerCount ?? game.players().length;
  const rank = context?.rankByID.get(actor.id()) ?? rankOf(game, actor);
  return [
    actor.hasSpawned() ? 1 : 0,
    actor.isAlive() ? 1 : 0,
    actor.numTilesOwned() / Math.max(1, game.numLandTiles()),
    actor.troops() / 10_000_000,
    Number(actor.gold()) / 10_000_000,
    rank / Math.max(1, playerCount),
    actor.borderTiles().size / Math.max(1, actor.numTilesOwned()),
    sumAttackTroops(actor.incomingAttacks()) / 10_000_000,
    sumAttackTroops(actor.outgoingAttacks()) / 10_000_000,
    actor.unitCount(UnitType.Warship) / 20,
    actor.unitCount(UnitType.Port) / 20,
    actor.unitCount(UnitType.MissileSilo) / 20,
  ];
}

function targetFeatures(
  game: GameRunner["game"],
  actor: Player | null,
  target: Player | null,
  context?: CandidateFeatureContext | null,
): number[] {
  if (target === null) return new Array(14).fill(0);
  const cached = context?.targetFeatureCache.get(target.id());
  if (cached !== undefined) return cached;
  const playerCount = context?.playerCount ?? game.players().length;
  const rank = context?.rankByID.get(target.id()) ?? rankOf(game, target);
  let contact = 0;
  if (actor !== null) {
    const cachedContact = context?.borderContactCache.get(target.id());
    if (cachedContact !== undefined) {
      contact = cachedContact;
    } else {
      contact = borderContact(game, actor, target);
      context?.borderContactCache.set(target.id(), contact);
    }
  }
  const features = [
    1,
    target.isAlive() ? 1 : 0,
    target.type() === PlayerType.Nation ? 1 : 0,
    context !== undefined && context !== null
      ? context.leaderID === target.id()
        ? 1
        : 0
      : leaderByTiles(game)?.id() === target.id()
        ? 1
        : 0,
    actor !== null && actor.sharesBorderWith(target) ? 1 : 0,
    target.numTilesOwned() / Math.max(1, game.numLandTiles()),
    target.troops() / 10_000_000,
    Number(target.gold()) / 10_000_000,
    rank / Math.max(1, playerCount),
    target.borderTiles().size / Math.max(1, target.numTilesOwned()),
    sumAttackTroops(target.incomingAttacks()) / 10_000_000,
    sumAttackTroops(target.outgoingAttacks()) / 10_000_000,
    actor !== null ? actor.troops() / Math.max(1, target.troops()) : 0,
    contact / Math.max(1, actor?.borderTiles().size ?? 1),
  ].map((value) => clamp(value, 0, 10));
  context?.targetFeatureCache.set(target.id(), features);
  return features;
}

function tileFeatures(
  game: GameRunner["game"],
  actor: Player | null,
  tile: TileRef | undefined,
  context?: CandidateFeatureContext | null,
): number[] {
  if (tile === undefined || !game.isValidRef(tile)) return new Array(15).fill(0);
  const cached = context?.tileFeatureCache.get(tile);
  if (cached !== undefined) return cached;
  const owner = game.hasOwner(tile) ? game.owner(tile) : null;
  const ownerIsActor =
    owner !== null && owner.isPlayer() && actor !== null && owner.id() === actor.id();
  const ownerIsEnemy =
    owner !== null &&
    owner.isPlayer() &&
    actor !== null &&
    owner.id() !== actor.id();
  const features = [
    1,
    game.isLand(tile) ? 1 : 0,
    game.isShore(tile) ? 1 : 0,
    game.hasFallout(tile) ? 1 : 0,
    owner === null ? 1 : 0,
    ownerIsActor ? 1 : 0,
    ownerIsEnemy ? 1 : 0,
    owner !== null && owner.isPlayer() && owner.type() === PlayerType.Nation
      ? 1
      : 0,
    owner !== null && owner.isPlayer()
      ? owner.numTilesOwned() / Math.max(1, game.numLandTiles())
      : 0,
    owner !== null && owner.isPlayer() ? owner.troops() / 10_000_000 : 0,
    normalizedNearestUnitDistance(game, tile, UnitType.Port, actor, true, context),
    normalizedNearestUnitDistance(game, tile, UnitType.Port, actor, false, context),
    normalizedNearestUnitDistance(game, tile, UnitType.Warship, actor, true, context),
    normalizedNearestUnitDistance(game, tile, UnitType.Warship, actor, false, context),
    nearbyUnitCount(game, tile, UnitType.Warship, actor, false, 20, context) / 10,
  ].map((value) => clamp(value, 0, 10));
  context?.tileFeatureCache.set(tile, features);
  return features;
}

function intentFeatures(
  game: GameRunner["game"],
  actor: Player | null,
  target: Player | null,
  tile: TileRef | undefined,
  intent?: Intent,
  context?: CandidateFeatureContext | null,
): number[] {
  const troopCount =
    intent !== undefined && "troops" in intent && typeof intent.troops === "number"
      ? intent.troops
      : 0;
  const troopFraction =
    actor !== null && troopCount > 0 ? troopCount / Math.max(1, actor.troops()) : 0;
  const targetValue =
    target !== null
      ? target.numTilesOwned() / Math.max(1, game.numLandTiles()) +
        target.troops() / 10_000_000
      : 0;
  const attackRisk =
    target !== null && actor !== null && troopCount > 0
      ? target.troops() / Math.max(1, troopCount)
      : 0;
  const navalRisk =
    tile !== undefined
      ? nearbyUnitCount(game, tile, UnitType.Warship, actor, false, 40, context) / 5
      : 0;
  return [
    troopFraction,
    targetValue,
    attackRisk,
    navalRisk,
    intent?.type === "attack" && target === null ? 1 : 0,
    intent?.type === "attack" && target !== null ? 1 : 0,
    intent?.type === "boat" ? 1 : 0,
    intent?.type === "build_unit" ? 1 : 0,
    intent?.type === "upgrade_structure" ? 1 : 0,
    tile !== undefined && actor !== null && game.isLand(tile) && !game.hasOwner(tile)
      ? 1
      : 0,
  ].map((value) => clamp(value, 0, 10));
}

const LOCAL_PATCH_GRID = 9;
const LOCAL_PATCH_FEATURES_PER_RADIUS = 29;

function localPatchFeatures(
  game: GameRunner["game"],
  actor: Player | null,
  tile: TileRef | undefined,
  radii: number[],
  context?: CandidateFeatureContext | null,
): number[] {
  const cleanRadii = radii
    .map((radius) => Math.max(1, Math.floor(radius)))
    .filter((radius, index, values) => values.indexOf(radius) === index);
  if (cleanRadii.length === 0) return [];
  if (tile === undefined || !game.isValidRef(tile)) {
    return new Array(cleanRadii.length * LOCAL_PATCH_FEATURES_PER_RADIUS).fill(0);
  }
  const cacheKey = `${tile}:${cleanRadii.join(",")}`;
  const cached = context?.localPatchFeatureCache.get(cacheKey);
  if (cached !== undefined) return cached;

  const result: number[] = [];
  const cx = game.x(tile);
  const cy = game.y(tile);
  for (const radius of cleanRadii) {
    result.push(...sampledLocalPatchFeatures(game, actor, cx, cy, radius));
  }
  context?.localPatchFeatureCache.set(cacheKey, result);
  return result;
}

function sampledLocalPatchFeatures(
  game: GameRunner["game"],
  actor: Player | null,
  cx: number,
  cy: number,
  radius: number,
): number[] {
  let valid = 0;
  let land = 0;
  let water = 0;
  let shore = 0;
  let unowned = 0;
  let own = 0;
  let enemy = 0;
  let nation = 0;
  let fallout = 0;
  let ownBorder = 0;
  let enemyBorder = 0;
  let ownTroops = 0;
  let enemyTroops = 0;

  for (let sy = 0; sy < LOCAL_PATCH_GRID; sy++) {
    for (let sx = 0; sx < LOCAL_PATCH_GRID; sx++) {
      const x = Math.round(
        cx - radius + (radius * 2 * sx) / Math.max(1, LOCAL_PATCH_GRID - 1),
      );
      const y = Math.round(
        cy - radius + (radius * 2 * sy) / Math.max(1, LOCAL_PATCH_GRID - 1),
      );
      if (!game.isValidCoord(x, y)) continue;
      const sample = game.ref(x, y);
      valid++;
      if (game.isLand(sample)) land++;
      if (game.isWater(sample)) water++;
      if (game.isShore(sample)) shore++;
      if (game.hasFallout(sample)) fallout++;
      if (!game.hasOwner(sample)) {
        if (game.isLand(sample)) unowned++;
        continue;
      }
      const owner = game.owner(sample);
      if (!owner.isPlayer()) continue;
      const troopsPerTile =
        owner.troops() / Math.max(1, owner.numTilesOwned()) / 1_000_000;
      if (actor !== null && owner.id() === actor.id()) {
        own++;
        ownTroops += troopsPerTile;
        if (owner.borderTiles().has(sample)) ownBorder++;
      } else {
        enemy++;
        enemyTroops += troopsPerTile;
        if (owner.borderTiles().has(sample)) enemyBorder++;
        if (owner.type() === PlayerType.Nation) nation++;
      }
    }
  }

  const attackSummary = localAttackSummary(game, actor, cx, cy, radius);
  const structureSummary = localStructureSummary(game, actor, cx, cy, radius);
  const denom = Math.max(1, valid);
  return [
    radius / Math.max(1, game.width() + game.height()),
    valid / (LOCAL_PATCH_GRID * LOCAL_PATCH_GRID),
    land / denom,
    water / denom,
    shore / denom,
    unowned / denom,
    own / denom,
    enemy / denom,
    nation / denom,
    fallout / denom,
    ownBorder / denom,
    enemyBorder / denom,
    ownTroops / denom,
    enemyTroops / denom,
    attackSummary.outgoingTroops / 10_000_000,
    attackSummary.incomingTroops / 10_000_000,
    attackSummary.outgoingPositions / 100,
    attackSummary.incomingPositions / 100,
    attackSummary.enemyAttackers / Math.max(1, game.players().length),
    structureSummary.ownPorts / 10,
    structureSummary.enemyPorts / 10,
    structureSummary.ownCities / 10,
    structureSummary.enemyCities / 10,
    structureSummary.ownSams / 10,
    structureSummary.enemySams / 10,
    structureSummary.ownMissileSilos / 10,
    structureSummary.enemyMissileSilos / 10,
    structureSummary.ownDefensePosts / 10,
    structureSummary.enemyDefensePosts / 10,
  ].map((value) => clamp(value, 0, 10));
}

function localStructureSummary(
  game: GameRunner["game"],
  actor: Player | null,
  cx: number,
  cy: number,
  radius: number,
): {
  ownPorts: number;
  enemyPorts: number;
  ownCities: number;
  enemyCities: number;
  ownSams: number;
  enemySams: number;
  ownMissileSilos: number;
  enemyMissileSilos: number;
  ownDefensePosts: number;
  enemyDefensePosts: number;
} {
  const summary = {
    ownPorts: 0,
    enemyPorts: 0,
    ownCities: 0,
    enemyCities: 0,
    ownSams: 0,
    enemySams: 0,
    ownMissileSilos: 0,
    enemyMissileSilos: 0,
    ownDefensePosts: 0,
    enemyDefensePosts: 0,
  };
  for (const unit of game.units()) {
    const type = unit.type();
    if (
      type !== UnitType.Port &&
      type !== UnitType.City &&
      type !== UnitType.SAMLauncher &&
      type !== UnitType.MissileSilo &&
      type !== UnitType.DefensePost
    ) {
      continue;
    }
    const tile = unit.tile();
    const dist = Math.abs(game.x(tile) - cx) + Math.abs(game.y(tile) - cy);
    if (dist > radius) continue;
    const own = actor !== null && unit.owner().id() === actor.id();
    if (type === UnitType.Port) {
      if (own) summary.ownPorts++;
      else summary.enemyPorts++;
    } else if (type === UnitType.City) {
      if (own) summary.ownCities++;
      else summary.enemyCities++;
    } else if (type === UnitType.SAMLauncher) {
      if (own) summary.ownSams++;
      else summary.enemySams++;
    } else if (type === UnitType.MissileSilo) {
      if (own) summary.ownMissileSilos++;
      else summary.enemyMissileSilos++;
    } else if (type === UnitType.DefensePost) {
      if (own) summary.ownDefensePosts++;
      else summary.enemyDefensePosts++;
    }
  }
  return summary;
}

function localAttackSummary(
  game: GameRunner["game"],
  actor: Player | null,
  cx: number,
  cy: number,
  radius: number,
): {
  outgoingTroops: number;
  incomingTroops: number;
  outgoingPositions: number;
  incomingPositions: number;
  enemyAttackers: number;
} {
  let outgoingTroops = 0;
  let incomingTroops = 0;
  let outgoingPositions = 0;
  let incomingPositions = 0;
  const enemyAttackers = new Set<PlayerID>();
  for (const player of game.players()) {
    for (const attack of player.outgoingAttacks()) {
      let nearbyPositions = 0;
      for (const position of attack.clusteredPositions()) {
        const dist =
          Math.abs(game.x(position) - cx) + Math.abs(game.y(position) - cy);
        if (dist <= radius) nearbyPositions++;
      }
      if (nearbyPositions === 0) continue;
      if (actor !== null && player.id() === actor.id()) {
        outgoingPositions += nearbyPositions;
        outgoingTroops += attack.troops();
      } else {
        incomingPositions += nearbyPositions;
        incomingTroops += attack.troops();
        enemyAttackers.add(player.id());
      }
    }
  }
  return {
    outgoingTroops,
    incomingTroops,
    outgoingPositions,
    incomingPositions,
    enemyAttackers: enemyAttackers.size,
  };
}

function targetPlayerForIntent(
  game: GameRunner["game"],
  intent: Intent,
): Player | null {
  const maybeIDs: unknown[] = [];
  if ("targetID" in intent) maybeIDs.push(intent.targetID);
  if ("recipient" in intent) maybeIDs.push(intent.recipient);
  if ("target" in intent) maybeIDs.push(intent.target);
  if ("requestor" in intent) maybeIDs.push(intent.requestor);
  for (const maybeID of maybeIDs) {
    if (typeof maybeID === "string" && game.hasPlayer(maybeID)) {
      return game.player(maybeID);
    }
  }
  return null;
}

function rankOf(game: GameRunner["game"], player: Player): number {
  const sorted = game
    .players()
    .filter((p) => p.isAlive())
    .sort((a, b) => b.numTilesOwned() - a.numTilesOwned());
  const rank = sorted.findIndex((p) => p.id() === player.id()) + 1;
  return rank <= 0 ? sorted.length : rank;
}

function leaderByTiles(game: GameRunner["game"]): Player | null {
  return (
    game
      .players()
      .filter((player) => player.isAlive())
      .sort((a, b) => b.numTilesOwned() - a.numTilesOwned())[0] ?? null
  );
}

function sumAttackTroops(attacks: Attack[]): number {
  return attacks.reduce((sum, attack) => sum + attack.troops(), 0);
}

function borderContact(
  game: GameRunner["game"],
  actor: Player,
  target: Player,
): number {
  let contact = 0;
  for (const tile of actor.borderTiles()) {
    for (const neighbor of game.neighbors(tile)) {
      if (game.hasOwner(neighbor)) {
        const owner = game.owner(neighbor);
        if (owner.isPlayer() && owner.id() === target.id()) contact++;
      }
    }
  }
  return contact;
}

function normalizedNearestUnitDistance(
  game: GameRunner["game"],
  tile: TileRef,
  type: UnitType,
  actor: Player | null,
  friendly: boolean,
  context?: CandidateFeatureContext | null,
): number {
  const cacheKey = `${tile}:${type}:${friendly ? "1" : "0"}`;
  const cached = context?.nearestUnitCache.get(cacheKey);
  if (cached !== undefined) return cached;
  let best = Number.POSITIVE_INFINITY;
  for (const unit of game.units(type)) {
    const isFriendly = actor !== null && unit.owner().id() === actor.id();
    if (isFriendly !== friendly) continue;
    const dist = Math.abs(game.x(tile) - game.x(unit.tile())) +
      Math.abs(game.y(tile) - game.y(unit.tile()));
    best = Math.min(best, dist);
  }
  const result = Number.isFinite(best)
    ? clamp(best / Math.max(1, game.width() + game.height()), 0, 1)
    : 1;
  context?.nearestUnitCache.set(cacheKey, result);
  return result;
}

function nearbyUnitCount(
  game: GameRunner["game"],
  tile: TileRef,
  type: UnitType,
  actor: Player | null,
  friendly: boolean,
  radius: number,
  context?: CandidateFeatureContext | null,
): number {
  const cacheKey = `${tile}:${type}:${friendly ? "1" : "0"}:${radius}`;
  const cached = context?.nearbyUnitCountCache.get(cacheKey);
  if (cached !== undefined) return cached;
  let count = 0;
  for (const unit of game.units(type)) {
    const isFriendly = actor !== null && unit.owner().id() === actor.id();
    if (isFriendly !== friendly) continue;
    const dist = Math.abs(game.x(tile) - game.x(unit.tile())) +
      Math.abs(game.y(tile) - game.y(unit.tile()));
    if (dist <= radius) count++;
  }
  context?.nearbyUnitCountCache.set(cacheKey, count);
  return count;
}

function hasLandBorderWithTerraNullius(
  game: GameRunner["game"],
  player: Player,
): boolean {
  for (const border of player.borderTiles()) {
    for (const neighbor of game.neighbors(border)) {
      if (game.isLand(neighbor) && !game.hasOwner(neighbor)) return true;
    }
  }
  return false;
}

function sampleEvenly<T>(items: T[], maxItems: number): T[] {
  if (items.length <= maxItems) return items;
  const result: T[] = [];
  const step = items.length / maxItems;
  for (let i = 0; i < maxItems; i++) {
    result.push(items[Math.floor(i * step)]);
  }
  return result;
}

function isWinnerForClient(winner: Winner, clientID: ClientID): boolean {
  return Array.isArray(winner) && winner.includes(clientID);
}

function formatOpponentCounts(counts: OpponentCounts): string {
  return `bots=${counts.bots}, nations=${counts.nations}`;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
