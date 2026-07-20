import path from "node:path";
import { performance } from "node:perf_hooks";
import { createHash } from "node:crypto";
import { deflateRawSync } from "node:zlib";
import quickChatData from "resources/QuickChat.json";
import { Config } from "../core/configuration/Config";
import { Executor } from "../core/execution/ExecutionManager";
import { NationExecution } from "../core/execution/NationExecution";
import { GameRunner } from "../core/GameRunner";
import {
  AllPlayers,
  Attack,
  Difficulty,
  Execution,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  Nation,
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
import {
  ErrorUpdate,
  GameUpdateType,
  GameUpdateViewData,
} from "../core/game/GameUpdates";
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
  urgentEventYield: boolean;
  donateGold: boolean;
  donateTroops: boolean;
  waterNukes: boolean;
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
  // TEST-ONLY sandbox knob (default false): lets build/legality contract
  // tests exercise the build path without waiting for the real gold curve.
  // Never set by training or evaluation configs.
  infiniteGold?: boolean;
  // Run the native nation strategy for the first controlled player. Used only
  // to extract policy-observation/native-action teacher pairs.
  nativeTeacher?: boolean;
}

export interface ActionCandidate {
  kind: CandidateKind;
  label: string;
  intent?: Intent;
  features: number[];
}

export interface ExecutionTranscriptPlayer {
  id: PlayerID;
  smallID: number;
  type: PlayerType;
  clientID: ClientID | null;
  displayName: string;
}

export interface ExecutionTranscriptEntry {
  tick: number;
  type: string;
  fields: Record<
    string,
    string | number | boolean | null | ExecutionTranscriptPlayer
  >;
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

const PRECISION_V3_SIZE = 128;
const PRECISION_V3_PLANES = [
  "ownFraction",
  "opponentFraction",
  "neutralFraction",
  "waterPresence",
  "landPresence",
  "highlandPresence",
  "mountainPresence",
  "shorePresence",
  "falloutPresence",
  "frontierPresence",
  "dominantOwnerSmallID",
] as const;
const PRECISION_V3_LOCAL_PLANES = [
  "valid",
  "land",
  "water",
  "highland",
  "mountain",
  "shore",
  "fallout",
  "neutral",
  "own",
  "opponent",
  "ownerSmallID",
] as const;
const PRECISION_V3_ENTITY_KINDS = [
  UnitType.City, UnitType.Port, UnitType.MissileSilo, UnitType.SAMLauncher,
  UnitType.DefensePost, UnitType.Factory, UnitType.Warship,
  UnitType.AtomBomb, UnitType.HydrogenBomb, UnitType.MIRV,
  UnitType.TransportShip, UnitType.Train,
] as const;
const PRECISION_V3_TRAIN_TYPES = ["none", "Engine", "TailEngine", "Carriage"] as const;
const PRECISION_V3_TRAIN_LOAD_STATES = ["none", "unloaded", "loaded"] as const;
const PRECISION_V3_STATION_STATES = ["absent", "present"] as const;
const PRECISION_V3_SCHEMA = JSON.stringify({
  version: "precision-v3",
  globalSize: PRECISION_V3_SIZE,
  globalPlanes: PRECISION_V3_PLANES,
  localPlanes: PRECISION_V3_LOCAL_PLANES,
  localRadius: 2,
  globalLayout: "plane-major-row-major",
  localLayout: "patch-major-plane-major-row-major",
  frontier: "sorted exact TileRef plus owner smallID",
  entityKinds: PRECISION_V3_ENTITY_KINDS,
  trainTypes: PRECISION_V3_TRAIN_TYPES,
  trainLoadStates: PRECISION_V3_TRAIN_LOAD_STATES,
  stationStates: PRECISION_V3_STATION_STATES,
  entityLearnedFields: ["kind", "trainType", "trainLoadState", "hasTrainStation"],
  entityExactResolverMetadata: ["id", "tile", "unitType"],
  trainVisibility: "client-visible UnitUpdate trainType/loaded and hasTrainStation only; no railroad path, destination, or planning state",
  visibilityLineage: "docs/STATE_ACTION_INVENTORY.md schema 2 human-visible adapter fields",
});
const PRECISION_V3_SCHEMA_HASH = createHash("sha256")
  .update(PRECISION_V3_SCHEMA)
  .digest("hex");

const TEMPORAL_V1_SCHEMA = JSON.stringify({
  version: "temporal-v1",
  visibilityLineage: "client-visible PlayerUpdate attacks, Unit/UnitIncoming, and involved-player diplomacy events; STATE_ACTION_INVENTORY schema 2",
  delivery: "monotonic sequence cursor; reads are idempotent; explicit acknowledgement",
  eventOrder: "tick, kind, identity",
  mapDimensions: "width and height accompany standalone observations",
  eventKinds: [
    "incoming_attack", "weapon_launch", "weapon_incoming",
    "alliance_request", "alliance_reply", "alliance_broken",
    "alliance_expired", "alliance_extension", "embargo_start", "embargo_stop",
    "target_player", "donation_gold", "donation_troops",
  ],
  weaponKinds: ["Atom Bomb", "Hydrogen Bomb", "MIRV", "MIRV Warhead"],
  eventFields: [
    "sequence:int", "tick:int", "kind:enum", "identity:string-metadata",
    "actorOwner:smallID+1", "targetOwner:smallID+1", "amount:display-quantized",
    "tile:TileRef-metadata|null", "targetTile:TileRef-metadata|null",
    "unitType:enum|null", "accepted:boolean|null",
  ],
  currentThreats: [
    "incomingAttacks:{identity,actorOwner,amount}",
    "incomingWeapons:{identity,actorOwner,unitType,tile,targetTile}",
  ],
  diplomacy: [
    "alliances:{identity,otherOwner,createdAt,expiresAt}",
    "incomingRequests:{identity,otherOwner,createdAt}",
    "outgoingRequests:{identity,otherOwner}",
    "embargoes:{otherOwner,direction}", "targets:ownerSlot[]",
  ],
  earlyYield: "opt-in urgentEventYield only: new incoming_attack or weapon_incoming ends online microtick loop; next response step runs one tick; default legacy cadence and replay unchanged",
  encoderKinds: [
    "incoming_attack", "weapon_launch", "weapon_incoming", "alliance_request",
    "alliance_reply", "alliance_broken", "alliance_expired", "alliance_extension",
    "embargo_start", "embargo_stop", "target_player", "donation_gold",
    "donation_troops", "current_incoming_attack", "current_incoming_weapon",
    "current_alliance", "current_incoming_request", "current_outgoing_request",
    "current_embargo", "current_embargo_incoming", "current_target",
  ],
  encoderFeatures: [
    "age", "amount", "tileX", "tileY", "hasTile", "targetX", "targetY",
    "hasTarget", "acceptedKnown", "accepted", "isRecentEvent",
  ],
});
const TEMPORAL_V1_SCHEMA_HASH = createHash("sha256")
  .update(TEMPORAL_V1_SCHEMA)
  .digest("hex");

const LEGAL_DOMAINS_V1_SCHEMA = JSON.stringify({
  version: "legal-domains-v1",
  order: [
    "noop", "spawn", "attack", "build_unit", "boat", "move_warship",
    "allianceRequest", "allianceExtension", "allianceReject", "breakAlliance",
    "embargo", "emoji", "cancel_attack", "cancel_boat", "delete_unit",
    "upgrade_structure", "targetPlayer", "donate_gold", "donate_troops",
    "quick_chat", "embargo_all",
  ],
  exactOptions: "each emitted option has complete identity/TileRef/player/direction arguments; targeted missiles emit both flight directions; boat destinations match the water pathfinder's source/target component pair; family coverage is declared; independent of maxActions",
  continuous: "attack/boat/donation expose display-safe normalized fractions; shared resolver clamps fractions before exact amount construction; direct stale/out-of-range intents reject; troop donations may clamp to private recipient capacity and report actual amount",
  coverage: {
    exhaustive: ["spawn", "attack", "boat", "move_warship", "cancel_attack", "cancel_boat", "upgrade_structure", "delete_unit", "diplomacy", "donations", "communication"],
    deterministicAnchors: ["build_unit: bounded config-declared land/Port/Warship probes plus sampled enemy territory/unit missile targets; Phase3 measures coverage"],
  },
  nonIntentUnits: ["Train: auto-spawned by Train Station execution; no player action row"],
  compactTiles: "spawn and boat destinations, plus each reachable warship water component, use exact row-major little-endian bitsets compressed with raw DEFLATE and base64; count=0 has canonical data=''; inflated count>0 bytes equal ceil(tileCount/8); warship units join only to their declared component and the resolver excludes each unit's current/patrol tiles",
  unsupported: [],
});
const LEGAL_DOMAINS_V1_SCHEMA_HASH = createHash("sha256")
  .update(LEGAL_DOMAINS_V1_SCHEMA)
  .digest("hex");

const LEAN_V3_SCHEMA = JSON.stringify({
  version: "lean-v3",
  fields: ["summary", "precisionV3", "temporalV1", "legalDomainsV1", "timings"],
  exclusion: "never calls legacy observe/spatialPlanes/generateCandidates",
});
const LEAN_V3_SCHEMA_HASH = createHash("sha256").update(LEAN_V3_SCHEMA).digest("hex");

type TemporalEventKind =
  | "incoming_attack"
  | "weapon_launch"
  | "weapon_incoming"
  | "alliance_request"
  | "alliance_reply"
  | "alliance_broken"
  | "alliance_expired"
  | "alliance_extension"
  | "embargo_start"
  | "embargo_stop"
  | "target_player"
  | "donation_gold"
  | "donation_troops";

interface TemporalEventV1 {
  sequence: number;
  tick: number;
  kind: TemporalEventKind;
  identity: string;
  actorOwner: number;
  targetOwner: number;
  amount: number;
  tile: number | null;
  targetTile: number | null;
  unitType: string | null;
  accepted: boolean | null;
}

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

interface PrecisionV3Cache {
  width: number;
  height: number;
  terrainVersion: number;
  cellByTile: Int32Array;
  landDenom: Uint32Array;
  staticPlanes: Uint8Array;
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
  urgentEventYield: false,
  donateGold: false,
  donateTroops: false,
  waterNukes: false,
  maxTurns: 18_000,
  spatialSize: 32,
  maxActions: 256,
  maxPlayers: 64,
  compactCandidates: false,
  compactSpatial: false,
  nativeTeacher: false,
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
const TARGETED_MISSILE_UNITS = new Set<UnitType>([
  UnitType.AtomBomb,
  UnitType.HydrogenBomb,
  UnitType.MIRV,
]);
const TARGET_PRESERVING_BUILD_UNITS = new Set<UnitType>([
  ...TARGETED_MISSILE_UNITS,
  UnitType.Warship,
]);
const PRECISION_CELL_SIDE = 128;
const PRECISION_CELL_PATCH_V1_SCHEMA_HASH =
  "522719ec99ef5a2a76d5f56e3586b104174fd839ccac9c0bfb69bffe95aae03b";
const SPATIAL_ACTION_CONTEXT_V1_SCHEMA_HASH =
  "b774ab4e7c286bc4fa65ea65288e18932d513965a41a1339a125b34c3ba89844";
const MAX_STRATEGIC_AGE_TICKS = 10;
const MAX_EXECUTION_TRANSCRIPT_ENTRIES = 10_000;
const TEMPORAL_WEAPON_UNITS = new Set<UnitType>([
  UnitType.AtomBomb,
  UnitType.HydrogenBomb,
  UnitType.MIRV,
  UnitType.MIRVWarhead,
]);
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
  private precisionV3Cache: PrecisionV3Cache | null = null;
  private spatialActionContextCache = new Map<ClientID, {
    tick: number;
    precisionV3: ReturnType<OpenFrontGymEnv["precisionObservationV3"]>;
  }>();
  private shoreLandTilesCache: TileRef[] | null = null;
  private shoreLandTilesCacheVersion: number | null = null;
  private waterTilesByComponentCache: Map<number, TileRef[]> | null = null;
  private waterComponentCacheVersion: number | null = null;
  private waterComponentBitsetsCache = new Map<number, {
    count: number; tileCount: number; dtype: "bitset-deflate-raw"; data: string;
  }>();
  private candidateTimings: Record<string, number> | null = null;
  private candidateFeatureContext: CandidateFeatureContext | null = null;
  private actionTranscript: Array<{ turn: number; candidate: ActionCandidate }> =
    [];
  private nativeExecutionTranscript: ExecutionTranscriptEntry[] = [];
  private temporalEventsByClient = new Map<ClientID, TemporalEventV1[]>();
  private temporalNextSequenceByClient = new Map<ClientID, number>();
  private temporalSeenAttackIDsByClient = new Map<ClientID, Set<string>>();
  private temporalSeenLaunchUnitIDsByClient = new Map<ClientID, Set<number>>();
  private temporalSeenIncomingUnitIDsByClient = new Map<ClientID, Set<number>>();
  private urgentResponsePendingByClient = new Set<ClientID>();
  private legacyCandidateGenerationCount = 0;
  private lastLegalDomainTimings: Record<string, number> = {};

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
    this.precisionV3Cache = null;
    this.spatialActionContextCache = new Map();
    this.shoreLandTilesCache = null;
    this.shoreLandTilesCacheVersion = null;
    this.waterTilesByComponentCache = null;
    this.waterComponentCacheVersion = null;
    this.waterComponentBitsetsCache = new Map();
    this.actionTranscript = [];
    this.nativeExecutionTranscript = [];
    this.temporalEventsByClient = new Map();
    this.temporalNextSequenceByClient = new Map();
    this.temporalSeenAttackIDsByClient = new Map();
    this.temporalSeenLaunchUnitIDsByClient = new Map();
    this.temporalSeenIncomingUnitIDsByClient = new Map();
    this.urgentResponsePendingByClient = new Set();
    this.legacyCandidateGenerationCount = 0;
    this.lastLegalDomainTimings = {};
    this.startedAt = Date.now();

    const gameStart = buildGameStartInfo(this.config, this.controlledClientIDs);
    this.gameStartInfo = gameStart;
    this.runner = null;
    this.runner = await createIsolatedGameRunner(
      gameStart,
      this.clientID,
      this.config.mapsRoot,
      (update) => {
        if ("errMsg" in update) {
          this.lastError = update;
        } else {
          this.lastUpdate = update;
          this.captureTemporalUpdate(update);
        }
      },
      this.config.winPercent,
      (tick, execution) => this.captureExecution(tick, execution),
      this.config.nativeTeacher ? this.clientID : undefined,
    );
    this.resetTemporalBuffers();
    const preSpawnStarted = performance.now();
    const preSpawnTurns = this.preSpawnOpponents();
    this.resetTemporalBuffers();
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
    this.precisionV3Cache = null;
    this.spatialActionContextCache = new Map();
    this.shoreLandTilesCache = null;
    this.shoreLandTilesCacheVersion = null;
    this.waterTilesByComponentCache = null;
    this.waterComponentCacheVersion = null;
    this.waterComponentBitsetsCache = new Map();
    this.actionTranscript = [];
    this.nativeExecutionTranscript = [];
    this.temporalEventsByClient = new Map();
    this.temporalNextSequenceByClient = new Map();
    this.temporalSeenAttackIDsByClient = new Map();
    this.temporalSeenLaunchUnitIDsByClient = new Map();
    this.temporalSeenIncomingUnitIDsByClient = new Map();
    this.urgentResponsePendingByClient = new Set();
    this.legacyCandidateGenerationCount = 0;
    this.lastLegalDomainTimings = {};
    this.startedAt = Date.now();
    this.gameStartInfo = info;
    const replayClientIDs = observeClientIDs && observeClientIDs.length > 0
      ? observeClientIDs
      : info.players
        .map((player) => player.clientID)
        .filter((id): id is ClientID => id !== null);
    this.controlledClientIDs = [...new Set(replayClientIDs)];
    if (this.controlledClientIDs.length === 0) {
      throw new Error("resetFromRecord requires at least one observable client ID");
    }
    this.clientID = this.controlledClientIDs[0];
    this.runner = null;
    this.runner = await createIsolatedGameRunner(
      info,
      this.clientID,
      this.config.mapsRoot,
      (update) => {
        if ("errMsg" in update) {
          this.lastError = update;
        } else {
          this.lastUpdate = update;
          this.captureTemporalUpdate(update);
        }
      },
      undefined,
      (tick, execution) => this.captureExecution(tick, execution),
    );
    this.resetTemporalBuffers();
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
  replayAdvance(ticks: number): number {
    this.requireRunner();
    let applied = 0;
    for (let i = 0; i < ticks; i++) {
      if (this.isDone()) break;
      this.enqueueTurn([]);
      this.runner!.executeNextTick();
      if (this.lastError !== null) throw new Error(this.lastError.errMsg);
      applied += 1;
    }
    return applied;
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

  /** Apply consecutive recorded turns without one JSON-RPC round trip per tick. */
  replayApplyBatch(turns: StampedIntent[][]): number {
    this.requireRunner();
    let applied = 0;
    for (const intents of turns) {
      if (this.isDone()) break;
      this.enqueueTurn(intents);
      this.runner!.executeNextTick();
      if (this.lastError !== null) throw new Error(this.lastError.errMsg);
      applied += 1;
    }
    return applied;
  }

  /** Switch an exact replay reconstruction to normal policy stepping. */
  beginReplayTakeover(
    cid: ClientID = this.clientID,
    config: Partial<HeadlessEpisodeConfig> = {},
  ) {
    const player = this.requireRunner().game.playerByClientID(cid);
    if (player === null || !player.hasSpawned() || !player.isAlive()) {
      throw new Error("replay takeover player must be spawned and alive");
    }
    this.config = { ...this.config, ...config, controlledPlayers: 1 };
    this.clientID = cid;
    this.controlledClientIDs = [cid];
    this.maxTurnNumber = this.turnNumber + this.config.maxTurns;
    this.previousMetrics = this.metrics(cid);
    this.previousMetricsByClient = new Map([[cid, this.previousMetrics]]);
    this.resetTemporalBuffers();
    return {
      clientID: cid,
      turnNumber: this.turnNumber,
      tick: this.requireRunner().game.ticks(),
      maxTurnNumber: this.maxTurnNumber,
    };
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
    retreating: boolean;
  }[] {
    const game = this.requireRunner().game;
    const me = game.playerByClientID(cid);
    if (me === null) return [];
    return me.outgoingAttacks().map((a) => ({
      id: a.id(),
      troops: humanVisibleNumberFloor(a.troops() / 10) * 10,
      targetID: a.target().isPlayer() ? (a.target() as Player).id() : null,
      retreating: a.retreating(),
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

  /** ALL engine-provided legal spawn candidate tiles (same source as
   * validSpawnTile). The caller snaps the policy's learned coordinate to the
   * NEAREST of these instead of silently taking the first. */
  validSpawnTiles(cid: ClientID = this.clientID): number[] {
    const tiles: number[] = [];
    for (const c of this.generateCandidates(cid)) {
      if (c.kind === "intent" && c.intent !== undefined &&
          (c.intent as { type?: string }).type === "spawn") {
        const tile = (c.intent as { tile?: number }).tile;
        if (tile !== undefined) tiles.push(tile);
      }
    }
    return tiles;
  }

  /** Legality facts for one decision of `cid`.
   *
   * EXACT (execution-grade, safe to enable in a sampling mask):
   *   - spawned/alive gates;
   *   - attackNeutral (hasLandBorderWithTerraNullius — the exact neutral-attack
   *     predicate validateRawIntent applies);
   *   - attackTargets (alive+spawned players that share a border AND pass
   *     canAttackPlayer — the exact targeted-attack predicate, tightened to
   *     bordering);
   *   - spawnTiles (pre-spawn only): the engine's own generated spawn intent
   *     candidates, each individually legal;
   *   - buildOptions (post-spawn only): the engine's own generated build
   *     intent candidates as exact unit+tile pairs — every pair has already
   *     passed canBuildUnitType AND canBuild (the tile is the engine-snapped
   *     legal placement).
   *
   * INFORMATIONAL ONLY (affordability/counts — NOT proof that any concrete
   * frozen argument would be accepted; do NOT enable action families from
   * these alone): buildableUnits, outgoingAttacks, transports, warships,
   * ownUnits, otherPlayersAlive. */
  legalActions(cid: ClientID = this.clientID): {
    spawned: boolean;
    alive: boolean;
    attackNeutral: boolean;
    attackTargets: string[];
    spawnTiles: number[];
    buildOptions: {
      unit: string;
      tile: number;
      rocketDirectionUp?: boolean;
      targetPlayerID?: string;
    }[];
    cancelAttackOptions: { attackID: string; targetID: string | null }[];
    upgradeOptions: { unit: string; unitId: number; tile: number }[];
    deleteOptions: { unitId: number; tile: number }[];
    buildableUnits: string[];
    outgoingAttacks: number;
    transports: number;
    warships: number;
    ownUnits: number;
    otherPlayersAlive: number;
  } {
    const game = this.requireRunner().game;
    const me = game.playerByClientID(cid);
    const empty = {
      spawned: false,
      alive: false,
      attackNeutral: false,
      attackTargets: [] as string[],
      spawnTiles: [] as number[],
      buildOptions: [] as {
        unit: string;
        tile: number;
        rocketDirectionUp?: boolean;
        targetPlayerID?: string;
      }[],
      cancelAttackOptions: [] as { attackID: string; targetID: string | null }[],
      upgradeOptions: [] as { unit: string; unitId: number; tile: number }[],
      deleteOptions: [] as { unitId: number; tile: number }[],
      buildableUnits: [] as string[],
      outgoingAttacks: 0,
      transports: 0,
      warships: 0,
      ownUnits: 0,
      otherPlayersAlive: 0,
    };
    if (me === null) return empty;
    const spawned = me.hasSpawned();
    const alive = me.isAlive();
    const attackTargets: string[] = [];
    const buildableUnits: string[] = [];
    const spawnTiles: number[] = [];
    const buildOptions: {
      unit: string;
      tile: number;
      rocketDirectionUp?: boolean;
      targetPlayerID?: string;
    }[] = [];
    const cancelAttackOptions: { attackID: string; targetID: string | null }[] = [];
    const upgradeOptions: { unit: string; unitId: number; tile: number }[] = [];
    const deleteOptions: { unitId: number; tile: number }[] = [];
    let attackNeutral = false;
    let otherPlayersAlive = 0;
    // Exact options come from the engine's OWN candidate generator: pre-spawn
    // it emits legal spawn intents; post-spawn it emits build intents whose
    // unit passed canBuildUnitType and whose tile is the canBuild-snapped
    // legal placement.
    for (const c of this.generateCandidates(cid)) {
      if (c.kind !== "intent" || c.intent === undefined) continue;
      const it = c.intent as {
        type?: string;
        unit?: string;
        tile?: number;
        rocketDirectionUp?: boolean;
      };
      if (it.type === "spawn" && it.tile !== undefined) {
        spawnTiles.push(it.tile);
      } else if (
        it.type === "build_unit" &&
        it.unit !== undefined &&
        it.tile !== undefined
      ) {
        buildOptions.push({
          unit: it.unit,
          tile: it.tile,
          ...(it.rocketDirectionUp === undefined
            ? {}
            : { rocketDirectionUp: it.rocketDirectionUp }),
        });
      }
    }
    if (spawned && alive) {
      attackNeutral = hasLandBorderWithTerraNullius(game, me);
      for (const p of game.players()) {
        if (p === me || !p.isAlive() || !p.hasSpawned()) continue;
        otherPlayersAlive++;
        // targeted attacks may only choose an attackable BORDERING player
        if (me.sharesBorderWith(p) && me.canAttackPlayer(p, true)) {
          attackTargets.push(p.id());
        }
      }
      for (const unitType of PlayerBuildable.types) {
        if (this.canBuildUnitType(me, unitType)) buildableUnits.push(unitType);
      }
      const seenBuildOptions = new Set(
        buildOptions.map((o) => `${o.unit}:${o.tile}`),
      );
      // Build placement legality must not disappear because attack/boat
      // candidates consumed the shared maxActions budget.
      for (const option of this.placementBuildOptions(me)) {
        const key = `${option.unit}:${option.tile}`;
        if (seenBuildOptions.has(key)) continue;
        seenBuildOptions.add(key);
        buildOptions.push(option);
      }
      // Structure placement probes are local. Missile targets are a separate
      // exact set: every visible enemy unit tile plus sixteen deterministic
      // territory samples per enemy. This keeps native target TileRefs without
      // letting maxActions truncate the high-impact weapon options.
      for (const target of this.missileTargetTiles(me)) {
        for (const unitType of TARGETED_MISSILE_UNITS) {
          if (!PlayerBuildable.has(unitType)) continue;
          if (!this.canBuildUnitType(me, unitType)) continue;
          if (me.canBuild(unitType, target.tile) === false) continue;
          const key = `${unitType}:${target.tile}`;
          if (seenBuildOptions.has(key)) continue;
          seenBuildOptions.add(key);
          buildOptions.push({
            unit: unitType,
            tile: target.tile,
            rocketDirectionUp: true,
            targetPlayerID: target.targetPlayerID,
          });
        }
      }
      for (const attack of me.outgoingAttacks()) {
        if (attack.retreating()) continue;
        cancelAttackOptions.push({
          attackID: attack.id(),
          targetID: attack.target().isPlayer()
            ? (attack.target() as Player).id()
            : null,
        });
      }
      const canDelete = me.canDeleteUnit();
      for (const unit of me.units()) {
        if (me.canUpgradeUnit(unit)) {
          upgradeOptions.push({
            unit: unit.type(),
            unitId: unit.id(),
            tile: unit.tile(),
          });
        }
        if (
          canDelete &&
          unit.isActive() &&
          !unit.isMarkedForDeletion() &&
          game.isLand(unit.tile()) &&
          game.owner(unit.tile()) === me
        ) {
          deleteOptions.push({ unitId: unit.id(), tile: unit.tile() });
        }
      }
    }
    cancelAttackOptions.sort((a, b) => a.attackID.localeCompare(b.attackID));
    upgradeOptions.sort((a, b) => a.unitId - b.unitId);
    deleteOptions.sort((a, b) => a.unitId - b.unitId);
    return {
      spawned,
      alive,
      attackNeutral,
      attackTargets,
      spawnTiles,
      buildOptions,
      cancelAttackOptions,
      upgradeOptions,
      deleteOptions,
      buildableUnits,
      outgoingAttacks: me.outgoingAttacks().length,
      transports: me.units(UnitType.TransportShip).length,
      warships: me.units(UnitType.Warship).length,
      ownUnits: me.units().length,
      otherPlayersAlive,
    };
  }

  legalDomainsV1(cid: ClientID = this.clientID) {
    const allStarted = performance.now();
    const game = this.requireRunner().game;
    const player = game.playerByClientID(cid);
    const empty = {
      version: "legal-domains-v1" as const,
      schemaHash: LEGAL_DOMAINS_V1_SCHEMA_HASH,
      tick: game.ticks(),
      width: game.width(),
      height: game.height(),
      spawned: false,
      alive: false,
      noopOptions: [{}],
      spawnTiles: { count: 0, tileCount: game.width() * game.height(),
                    dtype: "bitset-deflate-raw" as const, data: "" },
      attackOptions: [] as { targetID: string | null }[],
      attackTroops: { kind: "fraction" as const, min: 0.02, max: 1,
                      suggestedFraction: 0.7, base: "display_self_troops" as const },
      buildOptions: [] as { unit: string; tile: number; rocketDirectionUp: boolean; targetPlayerID?: string }[],
      boatTiles: { count: 0, tileCount: game.width() * game.height(),
                   dtype: "bitset-deflate-raw" as const, data: "" },
      boatTroops: { kind: "fraction" as const, min: 0.02, max: 1,
                    suggestedFraction: 0.3, base: "display_self_troops" as const },
      moveWarshipDomain: {
        units: [] as { unitID: number; component: number; tile: number; excludedTiles: number[] }[],
        components: [] as { component: number; tiles: { count: number; tileCount: number; dtype: "bitset-deflate-raw"; data: string } }[],
      },
      cancelAttackOptions: [] as { attackID: string }[],
      cancelBoatOptions: [] as { unitID: number; tile: number }[],
      upgradeOptions: [] as { unit: string; unitId: number; tile: number }[],
      deleteOptions: [] as { unitId: number; tile: number }[],
      allianceRequestOptions: [] as { recipient: string }[],
      allianceExtensionOptions: [] as { recipient: string }[],
      allianceRejectOptions: [] as { requestor: string }[],
      breakAllianceOptions: [] as { recipient: string }[],
      targetPlayerOptions: [] as { target: string }[],
      embargoOptions: [] as { targetID: string; action: "start" | "stop" }[],
      donateGoldOptions: [] as { recipient: string; amount: { kind: "fraction"; min: number; max: number; engineDefaultFraction: number; base: "display_self_gold" } }[],
      donateTroopsOptions: [] as { recipient: string; amount: { kind: "fraction"; min: number; max: number; engineDefaultFraction: number; base: "display_self_troops"; recipientCapacityClamp: true } }[],
      emojiOptions: [] as { recipient: string; emoji: number }[],
      quickChatOptions: [] as { recipient: string; quickChatKey: string; target?: string }[],
      embargoAllOptions: [] as { action: "start" | "stop" }[],
      unsupportedFamilies: [] as string[],
      partialFamilies: ["build_unit"],
      buildProposal: {
        probeLimit: this.config.buildProbeLimit,
        exhaustive: this.config.buildProbeLimit === "full",
      },
      nonIntentUnits: ["Train"],
    };
    if (player === null) {
      this.lastLegalDomainTimings = { totalMs: performance.now() - allStarted };
      return empty;
    }
    empty.spawned = player.hasSpawned();
    empty.alive = player.isAlive();
    if (!player.hasSpawned()) {
      const spawnStarted = performance.now();
      const tiles = this.unownedLandTiles()
        .filter((tile) => !game.hasFallout(tile))
        .sort((a, b) => a - b);
      const bits = new Uint8Array(Math.ceil((game.width() * game.height()) / 8));
      for (const tile of tiles) bits[tile >> 3] |= 1 << (tile & 7);
      empty.spawnTiles = {
        count: tiles.length,
        tileCount: game.width() * game.height(),
        dtype: "bitset-deflate-raw",
        data: tiles.length === 0
          ? ""
          : Buffer.from(deflateRawSync(bits, { level: 1 })).toString("base64"),
      };
      this.lastLegalDomainTimings = {
        spawnMs: performance.now() - spawnStarted,
        totalMs: performance.now() - allStarted,
      };
      return empty;
    }
    if (!player.isAlive()) {
      this.lastLegalDomainTimings = { totalMs: performance.now() - allStarted };
      return empty;
    }

    let sectionStarted = performance.now();
    if (hasLandBorderWithTerraNullius(game, player)) {
      empty.attackOptions.push({ targetID: null });
    }
    for (const target of game.players().slice().sort((a, b) => a.id().localeCompare(b.id()))) {
      if (target === player || !target.isAlive() || !target.hasSpawned()) continue;
      if (player.sharesBorderWith(target) && player.canAttackPlayer(target, true)) {
        empty.attackOptions.push({ targetID: target.id() });
      }
    }
    const attackMs = performance.now() - sectionStarted;

    sectionStarted = performance.now();
    const seenBuild = new Set<string>();
    for (const option of this.placementBuildProposalOptions(player)) {
      seenBuild.add(`${option.unit}:${option.tile}`);
      empty.buildOptions.push(option);
    }
    for (const target of this.missileTargetTiles(player)) {
      for (const unit of TARGETED_MISSILE_UNITS as Set<PlayerBuildableUnitType>) {
        if (!this.canBuildUnitType(player, unit) || player.canBuild(unit, target.tile) === false) continue;
        const key = `${unit}:${target.tile}`;
        if (seenBuild.has(key)) continue;
        seenBuild.add(key);
        empty.buildOptions.push({
          unit, tile: target.tile as number, rocketDirectionUp: true,
          targetPlayerID: target.targetPlayerID,
        });
      }
    }
    empty.buildOptions = empty.buildOptions.flatMap((option) =>
      TARGETED_MISSILE_UNITS.has(option.unit as PlayerBuildableUnitType)
        ? [{ ...option, rocketDirectionUp: false },
           { ...option, rocketDirectionUp: true }]
        : [option],
    );
    empty.buildOptions.sort((a, b) =>
      a.unit.localeCompare(b.unit) || a.tile - b.tile ||
      Number(a.rocketDirectionUp) - Number(b.rocketDirectionUp) ||
      String(a.targetPlayerID ?? "").localeCompare(String(b.targetPlayerID ?? "")),
    );
    const buildMs = performance.now() - sectionStarted;

    sectionStarted = performance.now();
    if (this.canBuildUnitType(player, UnitType.TransportShip) &&
        player.unitCount(UnitType.TransportShip) < game.config().boatMaxNumber()) {
      const candidates = this.shoreLandTiles()
        .filter((tile) => {
          if (!game.isLand(tile) || !game.isShore(tile)) return false;
          if (game.hasOwner(tile)) {
            const owner = game.owner(tile);
            if (owner === player || (owner.isPlayer() && !player.canAttackPlayer(owner))) {
              return false;
            }
          }
          return true;
        });
      const probes = this.config.boatDestinationProbeLimit === "full"
        ? candidates
        : sampleEvenly(candidates, this.config.boatDestinationProbeLimit);
      const destinations = probes
        .filter((tile) => player.canBuild(UnitType.TransportShip, tile) !== false)
        .sort((a, b) => a - b);
      empty.boatTiles = this.tileBitset(destinations);
    }
    const boatMs = performance.now() - sectionStarted;
    sectionStarted = performance.now();
    const warships = player.units(UnitType.Warship)
      .filter((warship) => warship.isActive())
      .sort((a, b) => a.id() - b.id());
    if (warships.length > 0) {
      const wantedComponents = new Set<number>();
      for (const unit of warships) {
        const component = game.getWaterComponent(unit.tile());
        if (component === null) continue;
        wantedComponents.add(component);
        const excludedTiles = new Set<number>([unit.tile()]);
        const patrolTile = unit.warshipState().patrolTile;
        if (patrolTile !== undefined) excludedTiles.add(patrolTile);
        empty.moveWarshipDomain.units.push({
          unitID: unit.id(), component, tile: unit.tile(),
          excludedTiles: [...excludedTiles].sort((a, b) => a - b),
        });
      }
      const waterByComponent = this.waterTilesByComponent();
      empty.moveWarshipDomain.components = [...wantedComponents]
        .sort((a, b) => a - b)
        .map((component) => {
          let tiles = this.waterComponentBitsetsCache.get(component);
          if (tiles === undefined) {
            tiles = this.tileBitset(waterByComponent.get(component) ?? []);
            this.waterComponentBitsetsCache.set(component, tiles);
          }
          return { component, tiles };
        });
    }
    const warshipMs = performance.now() - sectionStarted;
    sectionStarted = performance.now();
    empty.cancelAttackOptions = player.outgoingAttacks()
      .filter((attack) => !attack.retreating())
      .map((attack) => ({ attackID: attack.id() }))
      .sort((a, b) => a.attackID.localeCompare(b.attackID));
    empty.cancelBoatOptions = player.units(UnitType.TransportShip)
      .filter((unit) => !unit.transportShipState().isRetreating)
      .map((unit) => ({ unitID: unit.id(), tile: unit.tile() as number }))
      .sort((a, b) => a.unitID - b.unitID);
    for (const unit of player.units().slice().sort((a, b) => a.id() - b.id())) {
      if (player.canUpgradeUnit(unit)) {
        empty.upgradeOptions.push({ unit: unit.type(), unitId: unit.id(), tile: unit.tile() });
      }
      if (player.canDeleteUnit() && unit.isActive() && !unit.isMarkedForDeletion() &&
          game.isLand(unit.tile()) && game.owner(unit.tile()) === player) {
        empty.deleteOptions.push({ unitId: unit.id(), tile: unit.tile() });
      }
    }
    const idFamiliesMs = performance.now() - sectionStarted;

    sectionStarted = performance.now();
    for (const other of game.players().slice().sort((a, b) => a.id().localeCompare(b.id()))) {
      if (other === player || !other.isAlive()) continue;
      if (player.canSendAllianceRequest(other)) {
        empty.allianceRequestOptions.push({ recipient: other.id() });
      }
      const alliance = player.allianceWith(other);
      if (alliance !== null) {
        if (player.allianceInfo(other)?.canExtend === true) {
          empty.allianceExtensionOptions.push({ recipient: other.id() });
        }
        empty.breakAllianceOptions.push({ recipient: other.id() });
      }
      if (player.canTarget(other)) empty.targetPlayerOptions.push({ target: other.id() });
      empty.embargoOptions.push({
        targetID: other.id(), action: player.hasEmbargoAgainst(other) ? "stop" : "start",
      });
      if (player.canDonateGold(other)) {
        if (humanVisibleNumberFloor(Number(player.gold())) > 0) empty.donateGoldOptions.push({
          recipient: other.id(),
          amount: { kind: "fraction", min: 0, max: 1, engineDefaultFraction: 1 / 3,
                    base: "display_self_gold" },
        });
      }
      if (player.canDonateTroops(other)) {
        if (humanVisibleNumberFloor(player.troops() / 10) * 10 > 0) {
          empty.donateTroopsOptions.push({
          recipient: other.id(),
          amount: {
            kind: "fraction", min: 0, max: 1, engineDefaultFraction: 1 / 3,
            base: "display_self_troops", recipientCapacityClamp: true,
          },
        }); }
      }
      if (player.canSendEmoji(other)) {
        for (const emoji of EMOJIS) {
          empty.emojiOptions.push({ recipient: other.id(), emoji });
        }
      }
      if (player.canSendQuickChat(other)) {
        for (const entry of QUICK_CHAT_KEYS) {
          empty.quickChatOptions.push({
            recipient: other.id(),
            quickChatKey: entry.key,
            ...(entry.requiresPlayer ? { target: other.id() } : {}),
          });
        }
      }
    }
    if (player.canSendEmoji(AllPlayers)) {
      for (const emoji of EMOJIS) {
        empty.emojiOptions.push({ recipient: AllPlayers, emoji });
      }
    }
    if (player.canEmbargoAll()) {
      empty.embargoAllOptions.push({ action: "start" }, { action: "stop" });
    }
    empty.allianceRejectOptions = player.incomingAllianceRequests()
      .map((request) => ({ requestor: request.requestor().id() }))
      .sort((a, b) => a.requestor.localeCompare(b.requestor));
    this.lastLegalDomainTimings = {
      attackMs, buildMs, boatMs, warshipMs, idFamiliesMs,
      diplomacyMs: performance.now() - sectionStarted,
      totalMs: performance.now() - allStarted,
    };
    return empty;
  }

  canonicalBuildTile(
    cid: ClientID = this.clientID,
    unit: string,
    tile: number,
  ): number | null {
    const game = this.requireRunner().game;
    const player = game.playerByClientID(cid);
    if (player === null || !player.hasSpawned() || !player.isAlive() ||
        !PlayerBuildable.has(unit as PlayerBuildableUnitType) ||
        !game.isValidRef(tile) ||
        !this.canBuildUnitType(player, unit as PlayerBuildableUnitType)) {
      return null;
    }
    const canonical = player.canBuild(unit as PlayerBuildableUnitType, tile as TileRef);
    if (canonical === false) return null;
    return TARGET_PRESERVING_BUILD_UNITS.has(unit as PlayerBuildableUnitType)
      ? tile
      : canonical as number;
  }

  /** Human-visible native semantic pixels for one fixed 128x128 action cell. */
  precisionCellPatchV1(
    cid: ClientID = this.clientID,
    cellRow = 0,
    cellCol = 0,
  ) {
    const game = this.requireRunner().game;
    const me = game.playerByClientID(cid);
    const width = game.width();
    const height = game.height();
    const cellRows = Math.ceil(height / PRECISION_CELL_SIDE);
    const cellCols = Math.ceil(width / PRECISION_CELL_SIDE);
    if (!Number.isInteger(cellRow) || !Number.isInteger(cellCol) ||
        cellRow < 0 || cellRow >= cellRows || cellCol < 0 || cellCol >= cellCols) {
      throw new Error("precision-cell-patch-v1 coarse cell is outside the map");
    }
    const planes = [...PRECISION_V3_LOCAL_PLANES];
    const planeSize = PRECISION_CELL_SIDE * PRECISION_CELL_SIDE;
    const data = new Uint8Array(planes.length * planeSize);
    const set = (channel: number, localRow: number, localCol: number, value: number) => {
      data[channel * planeSize + localRow * PRECISION_CELL_SIDE + localCol] = value;
    };
    const startRow = cellRow * PRECISION_CELL_SIDE;
    const startCol = cellCol * PRECISION_CELL_SIDE;
    const validHeight = Math.min(PRECISION_CELL_SIDE, height - startRow);
    const validWidth = Math.min(PRECISION_CELL_SIDE, width - startCol);
    for (let localRow = 0; localRow < validHeight; localRow++) {
      for (let localCol = 0; localCol < validWidth; localCol++) {
        const tile = game.ref(startCol + localCol, startRow + localRow);
        set(0, localRow, localCol, 1);
        const land = game.isLand(tile);
        set(land ? 1 : 2, localRow, localCol, 1);
        if (game.terrainType(tile) === TerrainType.Highland) set(3, localRow, localCol, 1);
        if (game.terrainType(tile) === TerrainType.Mountain) set(4, localRow, localCol, 1);
        if (game.isShore(tile)) set(5, localRow, localCol, 1);
        if (game.hasFallout(tile)) set(6, localRow, localCol, 1);
        if (!game.hasOwner(tile)) {
          if (land) set(7, localRow, localCol, 1);
        } else {
          const owner = game.owner(tile);
          set(owner === me ? 8 : 9, localRow, localCol, 1);
          set(10, localRow, localCol, Math.min(255, owner.smallID() + 1));
        }
      }
    }
    return {
      version: "precision-cell-patch-v1" as const,
      schemaHash: PRECISION_CELL_PATCH_V1_SCHEMA_HASH,
      tick: game.ticks(), width, height, cellRow, cellCol, cellRows, cellCols,
      side: PRECISION_CELL_SIDE, validWidth, validHeight,
      planes, dtype: "uint8" as const,
      data: Buffer.from(data.buffer).toString("base64"),
    };
  }

  /** Fresh visible entities/economy plus an explicitly aged strategic snapshot. */
  private policyPlayers(cid: ClientID): Player[] {
    const game = this.requireRunner().game;
    const me = game.playerByClientID(cid);
    const selected: Player[] = [];
    const seen = new Set<PlayerID>();
    const add = (player: Player) => {
      if (seen.has(player.id())) return;
      seen.add(player.id());
      selected.push(player);
    };
    if (me !== null) {
      add(me);
      for (const attack of me.incomingAttacks()) add(attack.attacker());
      for (const attack of me.outgoingAttacks()) {
        const target = attack.target();
        if (target.isPlayer()) add(target);
      }
      for (const neighbor of me.nearby()) {
        if (neighbor.isPlayer()) add(neighbor);
      }
      for (const request of me.incomingAllianceRequests()) add(request.requestor());
      for (const request of me.outgoingAllianceRequests()) add(request.recipient());
      for (const ally of me.allies()) add(ally);
      for (const target of me.transitiveTargets()) add(target);
      for (const partner of me.tradingPartners()) add(partner);
      for (const embargo of me.getEmbargoes()) add(embargo.target);
    }
    for (const player of game.players().slice().sort((a, b) =>
      b.numTilesOwned() - a.numTilesOwned() || a.smallID() - b.smallID()
    )) {
      add(player);
    }
    return selected.slice(0, this.config.maxPlayers);
  }

  spatialActionContextV1(
    cid: ClientID = this.clientID,
    refreshStrategic = false,
    maxStrategicAgeTicks = 10,
    afterSequence = 0,
  ) {
    const currentTick = this.requireRunner().game.ticks();
    if (!Number.isInteger(maxStrategicAgeTicks) || maxStrategicAgeTicks < 0 ||
        maxStrategicAgeTicks > MAX_STRATEGIC_AGE_TICKS) {
      throw new Error(`maxStrategicAgeTicks must be an integer from 0 to ${MAX_STRATEGIC_AGE_TICKS}`);
    }
    const cached = this.spatialActionContextCache.get(cid);
    if (refreshStrategic || cached === undefined) {
      this.spatialActionContextCache.set(cid, {
        tick: currentTick,
        precisionV3: this.precisionObservationV3(cid),
      });
    }
    const strategic = this.spatialActionContextCache.get(cid)!;
    const strategicAgeTicks = currentTick - strategic.tick;
    const usable = strategicAgeTicks >= 0 && strategicAgeTicks <= maxStrategicAgeTicks;
    const game = this.requireRunner().game;
    const me = game.playerByClientID(cid);
    const metrics = this.metrics(cid);
    const visiblePlayers = this.policyPlayers(cid);
    const fresh = {
      vector: [
        metrics.myShare,
        metrics.myTiles / Math.max(1, game.numLandTiles()),
        (humanVisibleNumberFloor(metrics.troops / 10) * 10) / 10_000_000,
        humanVisibleNumberFloor(metrics.gold) / 10_000_000,
        metrics.rank / Math.max(1, game.players().length),
        metrics.enemiesAlive / Math.max(1, game.players().length - 1),
        game.inSpawnPhase() ? 1 : 0,
        this.isDone() ? 1 : 0,
      ],
      players: visiblePlayers
        .map((player) => summarizePlayer(player, me?.id() ?? null)),
      tokens: this.visibleEntityTokens(cid),
      temporalV1: this.temporalObservationV1(cid, afterSequence),
    };
    return {
      version: "spatial-action-context-v1" as const,
      schemaHash: SPATIAL_ACTION_CONTEXT_V1_SCHEMA_HASH,
      currentTick,
      strategicTick: strategic.tick,
      strategicAgeTicks,
      maxStrategicAgeTicks,
      strategicRefreshed: refreshStrategic || cached === undefined,
      usable,
      reason: usable ? null : "strategic_snapshot_stale",
      observation: fresh,
      strategicPrecisionV3: usable && (refreshStrategic || cached === undefined)
        ? strategic.precisionV3
        : null,
    };
  }

  /** Read-only exact dispatch gate. It reports canonicalization and rejects
   * stale observations without submitting an intent or exposing planning state. */
  validateSpatialIntentV1(
    cid: ClientID = this.clientID,
    intent: Intent,
    observedTick: number,
    strategicTick: number = observedTick,
    maxStrategicAgeTicks = 10,
  ) {
    const game = this.requireRunner().game;
    const currentTick = game.ticks();
    const stale = !Number.isInteger(observedTick) || observedTick !== currentTick;
    const cachedStrategic = this.spatialActionContextCache.get(cid);
    const invalidAgeLimit = !Number.isInteger(maxStrategicAgeTicks) ||
      maxStrategicAgeTicks < 0 || maxStrategicAgeTicks > MAX_STRATEGIC_AGE_TICKS;
    const missingStrategic = cachedStrategic === undefined;
    const strategicMismatch = !missingStrategic &&
      (!Number.isInteger(strategicTick) || strategicTick !== cachedStrategic.tick);
    const trustedStrategicTick = cachedStrategic?.tick ?? -1;
    const strategicAgeTicks = currentTick - trustedStrategicTick;
    const strategicStale = invalidAgeLimit || missingStrategic || strategicMismatch ||
      strategicAgeTicks < 0 || strategicAgeTicks > maxStrategicAgeTicks;
    const player = game.playerByClientID(cid);
    const result = this.validateRawIntent(intent, player);
    const legal = "intent" in result;
    const requestedTile = intent.type === "boat" ? intent.dst :
      intent.type === "spawn" || intent.type === "build_unit" ||
      intent.type === "move_warship" ? intent.tile : null;
    const canonicalTile = intent.type === "build_unit"
      ? this.canonicalBuildTile(cid, intent.unit, intent.tile)
      : legal ? requestedTile : null;
    return {
      version: "spatial-intent-validation-v1" as const,
      observedTick, currentTick, stale, strategicTick: trustedStrategicTick,
      requestedStrategicTick: strategicTick, strategicAgeTicks,
      maxStrategicAgeTicks, strategicStale, legal,
      dispatchable: legal && !stale && !strategicStale,
      rejected: stale || strategicStale || !legal,
      requestedTile,
      canonicalTile,
      canonicalized: canonicalTile !== null && canonicalTile !== requestedTile,
      reason: stale ? "stale_observation" : invalidAgeLimit
        ? "invalid_strategic_age_limit" : missingStrategic
        ? "missing_strategic_snapshot" : strategicMismatch
        ? "strategic_snapshot_mismatch" : strategicStale
        ? "strategic_snapshot_stale" : legal ? null :
        ("reason" in result ? result.reason : "invalid_intent"),
    };
  }

  leanObservationV3(
    cid: ClientID = this.clientID,
    afterSequence = 0,
    localTiles: number[] = [],
    localRadius = 2,
  ) {
    const started = performance.now();
    const beforeCandidates = this.legacyCandidateGenerationCount;
    const game = this.requireRunner().game;
    const me = game.playerByClientID(cid);
    const metrics = this.metrics(cid);
    const summaryStarted = performance.now();
    const players = this.policyPlayers(cid)
      .map((player) => summarizePlayer(player, me?.id() ?? null));
    const summary = {
      tick: game.ticks(), turnNumber: this.turnNumber,
      vector: [
        metrics.myShare,
        metrics.myTiles / Math.max(1, game.numLandTiles()),
        (humanVisibleNumberFloor(metrics.troops / 10) * 10) / 10_000_000,
        humanVisibleNumberFloor(metrics.gold) / 10_000_000,
        metrics.rank / Math.max(1, game.players().length),
        metrics.enemiesAlive / Math.max(1, game.players().length - 1),
        game.inSpawnPhase() ? 1 : 0,
        this.isDone() ? 1 : 0,
      ],
      players,
      tokens: this.visibleEntityTokens(cid),
    };
    const summaryMs = performance.now() - summaryStarted;
    const precisionStarted = performance.now();
    const precisionV3 = this.precisionObservationV3(cid, localTiles, localRadius);
    const precisionMs = performance.now() - precisionStarted;
    const temporalStarted = performance.now();
    const temporalV1 = this.temporalObservationV1(cid, afterSequence);
    const temporalMs = performance.now() - temporalStarted;
    const legalStarted = performance.now();
    const legalDomainsV1 = this.legalDomainsV1(cid);
    const legalMs = performance.now() - legalStarted;
    return {
      version: "lean-v3" as const,
      schemaHash: LEAN_V3_SCHEMA_HASH,
      summary,
      precisionV3,
      temporalV1,
      legalDomainsV1,
      timings: {
        summaryMs, precisionMs, temporalMs, legalMs,
        legalBreakdownMs: { ...this.lastLegalDomainTimings },
        totalMs: performance.now() - started,
        legacyCandidateCalls: this.legacyCandidateGenerationCount - beforeCandidates,
      },
    };
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
    opts: {
      spatial2?: boolean; precisionV3?: boolean; localTiles?: number[];
      localRadius?: number; temporalV1?: boolean; afterSequence?: number;
    } = {},
  ): {
    vector: number[];
    players: PlayerSummary[];
    tokens: Array<{
      kind: number; owner: number; rel: number;
      x: number; y: number; troops: number; health: number;
      tile: number; id: number; unitType: string; level: number;
      active: boolean; underConstruction: boolean;
      targetTile: number | null; markedForDeletion: boolean;
      trainType?: string | null; loaded?: boolean | null;
      hasTrainStation?: boolean;
    }>;
    spatial2?: {
      planes: string[];
      size: number;
      dtype: "uint8";
      data: string;
    };
    precisionV3?: ReturnType<OpenFrontGymEnv["precisionObservationV3"]>;
    temporalV1?: ReturnType<OpenFrontGymEnv["temporalObservationV1"]>;
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
      if (!u.isActive() && !u.isUnderConstruction()) continue;
      const tile = u.tile();
      const owner = u.owner();
      const isMe = me !== null && owner === me;
      tokens.push({
        kind: kindIndex.get(u.type()) ?? 0,
        owner: owner.smallID(),
        rel: isMe ? 0 : owner.isPlayer() ? 2 : 1, // self / nation-bot / player
        x: game.x(tile) / w,
        y: game.y(tile) / h,
        troops: Math.log1p(
          humanVisibleNumberFloor(Math.max(0, u.troops?.() ?? 0) / 10) * 10,
        ),
        health: Math.log1p(Math.max(0, Number(u.health?.() ?? 0))),
        tile: tile as number,
        // id/unitType are NOT part of the model observation (the encoder reads
        // only the fields above) — they let the intent translator resolve
        // "policy coordinate -> concrete unit" for move_warship / upgrade /
        // cancel / delete intents, which the engine validates by unit id.
        id: u.id(),
        unitType: u.type(),
        level: u.level(),
        active: u.isActive(),
        underConstruction: u.isUnderConstruction(),
        targetTile: TARGET_PRESERVING_BUILD_UNITS.has(u.type())
          ? (u.targetTile() ?? null)
          : null,
        markedForDeletion: u.isMarkedForDeletion(),
      });
    }
    const obs = this.observe(cid).observation;
    const result: {
      vector: number[];
      players: PlayerSummary[];
      tokens: Array<(typeof tokens)[number] & {
        trainType?: string | null; loaded?: boolean | null;
        hasTrainStation?: boolean;
      }>;
      spatial2?: {
        planes: string[];
        size: number;
        dtype: "uint8";
        data: string;
      };
      precisionV3?: ReturnType<OpenFrontGymEnv["precisionObservationV3"]>;
      temporalV1?: ReturnType<OpenFrontGymEnv["temporalObservationV1"]>;
    } = {
      vector: obs.vector,
      players: obs.players,
      tokens: opts.precisionV3 ? this.visibleEntityTokens(cid) : tokens,
    };
    if (opts.spatial2) {
      result.spatial2 = this.retinaPlanes(cid);
    }
    if (opts.precisionV3) {
      result.precisionV3 = this.precisionObservationV3(
        cid,
        opts.localTiles ?? [],
        opts.localRadius ?? 2,
      );
    }
    if (opts.temporalV1) {
      result.temporalV1 = this.temporalObservationV1(cid, opts.afterSequence ?? 0);
    }
    return result;
  }

  /** Idempotent cursor read. Events remain buffered until explicit ack; an
   * observation never consumes them, so multiple callers cannot steal cues. */
  temporalObservationV1(cid: ClientID = this.clientID, afterSequence = 0) {
    const game = this.requireRunner().game;
    const latestIssued = (this.temporalNextSequenceByClient.get(cid) ?? 1) - 1;
    if (!Number.isInteger(afterSequence) || afterSequence < 0 || afterSequence > latestIssued) {
      throw new Error(`invalid temporal-v1 cursor ${afterSequence}; latest is ${latestIssued}`);
    }
    const me = game.playerByClientID(cid);
    const events = (this.temporalEventsByClient.get(cid) ?? [])
      .filter((event) => event.sequence > afterSequence);
    if (me === null) {
      return {
        version: "temporal-v1" as const,
        schemaHash: TEMPORAL_V1_SCHEMA_HASH,
        tick: game.ticks(),
        width: game.width(),
        height: game.height(),
        latestSequence: latestIssued,
        events,
        current: { incomingAttacks: [], incomingWeapons: [] },
        diplomacy: {
          alliances: [], incomingRequests: [], outgoingRequests: [],
          embargoes: [], targets: [],
        },
      };
    }
    const ownerSlot = (player: Player) => player.smallID() + 1;
    const incomingAttacks = me.incomingAttacks()
      .map((attack) => ({
        identity: attack.id(),
        actorOwner: ownerSlot(attack.attacker()),
        amount: humanVisibleNumberFloor(Math.max(0, attack.troops()) / 10) * 10,
      }))
      .sort((a, b) => a.identity.localeCompare(b.identity));
    const incomingWeapons = game.units(...TEMPORAL_WEAPON_UNITS)
      .filter((unit) => {
        const target = unit.targetTile();
        return unit.isActive() && target !== undefined && game.hasOwner(target) &&
          game.owner(target) === me;
      })
      .map((unit) => ({
        identity: String(unit.id()),
        actorOwner: ownerSlot(unit.owner()),
        unitType: unit.type(),
        tile: unit.tile() as number,
        targetTile: unit.targetTile() as number,
      }))
      .sort((a, b) => Number(a.identity) - Number(b.identity));
    const alliances = me.alliances()
      .map((alliance) => ({
        identity: String(alliance.id()),
        otherOwner: ownerSlot(alliance.other(me)),
        createdAt: alliance.createdAt(),
        expiresAt: alliance.expiresAt(),
      }))
      .sort((a, b) => a.otherOwner - b.otherOwner || a.identity.localeCompare(b.identity));
    const incomingRequests = me.incomingAllianceRequests()
      .map((request) => ({
        identity: `${request.requestor().smallID()}:${request.createdAt()}`,
        otherOwner: ownerSlot(request.requestor()),
        createdAt: request.createdAt(),
      }))
      .sort((a, b) => a.otherOwner - b.otherOwner || a.createdAt - b.createdAt);
    const outgoingRequests = me.outgoingAllianceRequests()
      .map((request) => ({
        identity: String(request.recipient().smallID()),
        otherOwner: ownerSlot(request.recipient()),
      }))
      .sort((a, b) => a.otherOwner - b.otherOwner);
    const embargoes = [
      ...me.getEmbargoes().map((embargo) => ({
        otherOwner: ownerSlot(embargo.target),
        direction: "outgoing" as const,
      })),
      ...game.players()
        .filter((player) => player !== me && player.hasEmbargoAgainst(me))
        .map((player) => ({
          otherOwner: ownerSlot(player),
          direction: "incoming" as const,
        })),
    ]
      .sort((a, b) => a.otherOwner - b.otherOwner);
    const targets = [...new Set(me.transitiveTargets().map(ownerSlot))].sort((a, b) => a - b);
    return {
      version: "temporal-v1" as const,
      schemaHash: TEMPORAL_V1_SCHEMA_HASH,
      tick: game.ticks(),
      width: game.width(),
      height: game.height(),
      latestSequence: latestIssued,
      events,
      current: { incomingAttacks, incomingWeapons },
      diplomacy: { alliances, incomingRequests, outgoingRequests, embargoes, targets },
    };
  }

  acknowledgeTemporalEvents(
    cid: ClientID = this.clientID,
    throughSequence: number,
  ): number {
    const events = this.temporalEventsByClient.get(cid) ?? [];
    const latestIssued = (this.temporalNextSequenceByClient.get(cid) ?? 1) - 1;
    if (!Number.isInteger(throughSequence) || throughSequence < 0 ||
        throughSequence > latestIssued) {
      throw new Error(
        `invalid temporal-v1 acknowledgement ${throughSequence}; latest is ${latestIssued}`,
      );
    }
    const remaining = events.filter((event) => event.sequence > throughSequence);
    this.temporalEventsByClient.set(cid, remaining);
    return remaining.length;
  }

  private resetTemporalBuffers(): void {
    if (this.runner === null) return;
    const game = this.runner.game;
    const existingWeaponIDs = new Set(
      game.units(...TEMPORAL_WEAPON_UNITS).map((unit) => unit.id()),
    );
    this.urgentResponsePendingByClient.clear();
    for (const cid of this.controlledClientIDs) {
      const me = game.playerByClientID(cid);
      this.temporalEventsByClient.set(cid, []);
      this.temporalNextSequenceByClient.set(cid, 1);
      this.temporalSeenAttackIDsByClient.set(
        cid,
        new Set(me?.incomingAttacks().map((attack) => attack.id()) ?? []),
      );
      this.temporalSeenLaunchUnitIDsByClient.set(cid, new Set(existingWeaponIDs));
      this.temporalSeenIncomingUnitIDsByClient.set(cid, new Set());
    }
  }

  private captureTemporalUpdate(update: GameUpdateViewData): void {
    if (this.runner === null || this.temporalEventsByClient.size === 0) return;
    const game = this.runner.game;
    for (const cid of this.controlledClientIDs) {
      const me = game.playerByClientID(cid);
      if (me === null) continue;
      const selfSmallID = me.smallID();
      const seenAttacks = this.temporalSeenAttackIDsByClient.get(cid)!;
      const seenLaunches = this.temporalSeenLaunchUnitIDsByClient.get(cid)!;
      const seenIncoming = this.temporalSeenIncomingUnitIDsByClient.get(cid)!;
      const pending: Omit<TemporalEventV1, "sequence">[] = [];
      const base = (
        kind: TemporalEventKind,
        identity: string,
        actorOwner = 0,
        targetOwner = 0,
      ): Omit<TemporalEventV1, "sequence"> => ({
        tick: update.tick, kind, identity, actorOwner, targetOwner,
        amount: 0, tile: null, targetTile: null, unitType: null, accepted: null,
      });

      for (const playerUpdate of update.updates[GameUpdateType.Player]) {
        if (playerUpdate.id !== me.id() || playerUpdate.incomingAttacks === undefined) continue;
        for (const attack of playerUpdate.incomingAttacks) {
          if (seenAttacks.has(attack.id)) continue;
          seenAttacks.add(attack.id);
          const event = base(
            "incoming_attack", attack.id, attack.attackerID + 1, selfSmallID + 1,
          );
          event.amount = humanVisibleNumberFloor(Math.max(0, attack.troops) / 10) * 10;
          pending.push(event);
        }
      }
      for (const unitUpdate of update.updates[GameUpdateType.Unit]) {
        if (!TEMPORAL_WEAPON_UNITS.has(unitUpdate.unitType) || !unitUpdate.isActive ||
            seenLaunches.has(unitUpdate.id)) continue;
        seenLaunches.add(unitUpdate.id);
        const targetOwner = unitUpdate.targetTile !== undefined &&
          game.hasOwner(unitUpdate.targetTile)
          ? game.owner(unitUpdate.targetTile).smallID() + 1
          : 0;
        const event = base(
          "weapon_launch", String(unitUpdate.id), unitUpdate.ownerID + 1, targetOwner,
        );
        event.tile = unitUpdate.pos as number;
        event.targetTile = unitUpdate.targetTile ?? null;
        event.unitType = unitUpdate.unitType;
        pending.push(event);
      }
      for (const incoming of update.updates[GameUpdateType.UnitIncoming]) {
        if (incoming.playerID !== selfSmallID || seenIncoming.has(incoming.unitID)) continue;
        const unit = game.unit(incoming.unitID);
        if (unit === undefined || !TEMPORAL_WEAPON_UNITS.has(unit.type())) continue;
        seenIncoming.add(incoming.unitID);
        const event = base(
          "weapon_incoming", String(incoming.unitID),
          unit.owner().smallID() + 1,
          selfSmallID + 1,
        );
        event.tile = unit.tile() as number;
        event.targetTile = unit.targetTile() ?? null;
        event.unitType = unit.type();
        pending.push(event);
      }
      for (const request of update.updates[GameUpdateType.AllianceRequest]) {
        if (request.recipientID !== selfSmallID) continue;
        pending.push(base(
          "alliance_request", `${request.requestorID}:${request.createdAt}`,
          request.requestorID + 1, selfSmallID + 1,
        ));
      }
      for (const reply of update.updates[GameUpdateType.AllianceRequestReply]) {
        const request = reply.request;
        if (request.requestorID !== selfSmallID && request.recipientID !== selfSmallID) continue;
        const event = base(
          "alliance_reply", `${request.requestorID}:${request.recipientID}:${request.createdAt}`,
          request.recipientID + 1, request.requestorID + 1,
        );
        event.accepted = reply.accepted;
        pending.push(event);
      }
      for (const broken of update.updates[GameUpdateType.BrokeAlliance]) {
        if (broken.traitorID !== selfSmallID && broken.betrayedID !== selfSmallID) continue;
        pending.push(base(
          "alliance_broken", String(broken.allianceID),
          broken.traitorID + 1, broken.betrayedID + 1,
        ));
      }
      for (const expired of update.updates[GameUpdateType.AllianceExpired]) {
        if (expired.player1ID !== selfSmallID && expired.player2ID !== selfSmallID) continue;
        pending.push(base(
          "alliance_expired", `${expired.player1ID}:${expired.player2ID}`,
          expired.player1ID + 1, expired.player2ID + 1,
        ));
      }
      const allianceIDs = new Set(me.alliances().map((alliance) => alliance.id()));
      for (const extension of update.updates[GameUpdateType.AllianceExtension]) {
        if (extension.playerID !== selfSmallID || !allianceIDs.has(extension.allianceID)) continue;
        pending.push(base(
          "alliance_extension", `${extension.allianceID}:${extension.playerID}`,
          extension.playerID + 1, selfSmallID + 1,
        ));
      }
      for (const embargo of update.updates[GameUpdateType.EmbargoEvent]) {
        if (embargo.playerID !== selfSmallID && embargo.embargoedID !== selfSmallID) continue;
        pending.push(base(
          embargo.event === "start" ? "embargo_start" : "embargo_stop",
          `${embargo.playerID}:${embargo.embargoedID}:${embargo.event}`,
          embargo.playerID + 1, embargo.embargoedID + 1,
        ));
      }
      for (const target of update.updates[GameUpdateType.TargetPlayer]) {
        const sender = game.playerBySmallID(target.playerID);
        if (!sender.isPlayer() || !me.isFriendly(sender)) continue;
        pending.push(base(
          "target_player", `${target.playerID}:${target.targetID}`,
          target.playerID + 1, target.targetID + 1,
        ));
      }
      for (const donation of update.updates[GameUpdateType.DonateEvent]) {
        if (donation.senderId !== me.id() && donation.recipientId !== me.id()) continue;
        const sender = game.player(donation.senderId);
        const recipient = game.player(donation.recipientId);
        const event = base(
          donation.donationType === "gold" ? "donation_gold" : "donation_troops",
          `${donation.senderId}:${donation.recipientId}:${update.tick}:${donation.donationType}`,
          sender.smallID() + 1, recipient.smallID() + 1,
        );
        event.amount = donation.donationType === "gold"
          ? humanVisibleNumberFloor(Number(donation.amount))
          : humanVisibleNumberFloor(Number(donation.amount) / 10) * 10;
        pending.push(event);
      }
      pending.sort((a, b) =>
        a.tick - b.tick || a.kind.localeCompare(b.kind) || a.identity.localeCompare(b.identity),
      );
      const events = this.temporalEventsByClient.get(cid)!;
      let sequence = this.temporalNextSequenceByClient.get(cid)!;
      for (const event of pending) {
        events.push({ sequence: sequence++, ...event });
        if (event.kind === "incoming_attack" || event.kind === "weapon_incoming") {
          this.urgentResponsePendingByClient.add(cid);
        }
      }
      this.temporalNextSequenceByClient.set(cid, sequence);
    }
  }

  /** Human-visible precision-v3 semantic state. The global 128x128 map keeps
   * strategic geometry; exact, sorted frontier TileRefs and caller-requested
   * native patches preserve targeting precision without transferring the full
   * native ownership grid every decision. */
  precisionObservationV3(
    cid: ClientID = this.clientID,
    localTiles: number[] = [],
    localRadius = 2,
  ): {
    version: "precision-v3";
    schemaHash: string;
    width: number;
    height: number;
    global: { planes: string[]; size: number; dtype: "uint8"; data: string };
    frontier: {
      count: number;
      tileDtype: "int32-le";
      ownerDtype: "uint8";
      tiles: string;
      owners: string;
    };
    local: {
      radius: number;
      size: number;
      count: number;
      planes: string[];
      dtype: "uint8";
      centerTileDtype: "int32-le";
      centerTiles: string;
      data: string;
    };
  } {
    const game = this.requireRunner().game;
    const me = game.playerByClientID(cid);
    const width = game.width();
    const height = game.height();
    const size = PRECISION_V3_SIZE;
    const cellCount = size * size;
    const tileCount = width * height;
    const cache = this.getPrecisionV3Cache();
    const global = cache.staticPlanes.slice();
    const landDenom = cache.landDenom;
    const own = new Uint32Array(cellCount);
    const opponent = new Uint32Array(cellCount);
    const dominantCount = new Uint32Array(cellCount);
    const dominantOwner = new Uint8Array(cellCount);
    const cellByTile = cache.cellByTile;
    const ownerCellCount = new Uint32Array(cellCount);
    const touchedCells: number[] = [];
    const plane = (name: (typeof PRECISION_V3_PLANES)[number]) =>
      PRECISION_V3_PLANES.indexOf(name) * cellCount;

    if (game.numTilesWithFallout() > 0) {
      game.forEachTile((tile) => {
        if (game.hasFallout(tile)) {
          global[plane("falloutPresence") + cellByTile[tile]] = 255;
        }
      });
    }

    const frontierRows: Array<{ tile: number; owner: number }> = [];
    for (const owner of game.players()) {
      touchedCells.length = 0;
      const isSelf = owner.id() === me?.id();
      for (const tile of owner.tiles()) {
        const cell = cellByTile[tile];
        if (ownerCellCount[cell]++ === 0) touchedCells.push(cell);
        if (isSelf) own[cell]++;
        else opponent[cell]++;
      }
      for (const cell of touchedCells) {
        if (ownerCellCount[cell] > dominantCount[cell]) {
          dominantCount[cell] = ownerCellCount[cell];
          dominantOwner[cell] = Math.min(255, owner.smallID() + 1);
        }
        ownerCellCount[cell] = 0;
      }
      for (const tile of owner.borderTiles()) {
        frontierRows.push({ tile: tile as number, owner: owner.smallID() });
        global[plane("frontierPresence") + cellByTile[tile]] = 255;
      }
    }
    frontierRows.sort((a, b) => a.tile - b.tile || a.owner - b.owner);

    for (let cell = 0; cell < cellCount; cell++) {
      const d = Math.max(1, landDenom[cell]);
      global[plane("ownFraction") + cell] = Math.round((own[cell] / d) * 255);
      global[plane("opponentFraction") + cell] = Math.round((opponent[cell] / d) * 255);
      const neutral = Math.max(0, landDenom[cell] - own[cell] - opponent[cell]);
      global[plane("neutralFraction") + cell] = Math.round((neutral / d) * 255);
      global[plane("dominantOwnerSmallID") + cell] = dominantOwner[cell];
    }

    const frontierTiles = new Int32Array(frontierRows.map((row) => row.tile));
    const frontierOwners = new Uint8Array(frontierRows.map((row) => row.owner));
    if (localRadius !== 2) {
      throw new Error("precision-v3 localRadius is fixed at 2");
    }
    const radius = 2;
    const patchSize = radius * 2 + 1;
    const localPlaneSize = patchSize * patchSize;
    const local = new Uint8Array(
      localTiles.length * PRECISION_V3_LOCAL_PLANES.length * localPlaneSize,
    );
    const localPlane = (patch: number, channel: number) =>
      (patch * PRECISION_V3_LOCAL_PLANES.length + channel) * localPlaneSize;
    localTiles.forEach((centerValue, patch) => {
      const center = Math.floor(centerValue);
      if (centerValue !== center || center < 0 || center >= tileCount) {
        throw new Error(`invalid precision-v3 local center TileRef ${centerValue}`);
      }
      const cx = game.x(center as TileRef);
      const cy = game.y(center as TileRef);
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const px = cx + dx;
          const py = cy + dy;
          const at = (dy + radius) * patchSize + dx + radius;
          if (px < 0 || px >= width || py < 0 || py >= height) continue;
          const tile = game.ref(px, py);
          const set = (channel: number, value: number) => {
            local[localPlane(patch, channel) + at] = value;
          };
          set(0, 1);
          const land = game.isLand(tile);
          set(land ? 1 : 2, 1);
          if (game.terrainType(tile) === TerrainType.Highland) set(3, 1);
          if (game.terrainType(tile) === TerrainType.Mountain) set(4, 1);
          if (game.isShore(tile)) set(5, 1);
          if (game.hasFallout(tile)) set(6, 1);
          if (!game.hasOwner(tile)) {
            if (land) set(7, 1);
          } else {
            const owner = game.owner(tile);
            set(owner === me ? 8 : 9, 1);
            set(10, Math.min(255, owner.smallID() + 1));
          }
        }
      }
    });
    const centers = Int32Array.from(localTiles.map((tile) => Math.floor(tile)));

    return {
      version: "precision-v3",
      schemaHash: PRECISION_V3_SCHEMA_HASH,
      width,
      height,
      global: {
        planes: [...PRECISION_V3_PLANES],
        size,
        dtype: "uint8",
        data: Buffer.from(global.buffer).toString("base64"),
      },
      frontier: {
        count: frontierRows.length,
        tileDtype: "int32-le",
        ownerDtype: "uint8",
        tiles: Buffer.from(frontierTiles.buffer).toString("base64"),
        owners: Buffer.from(frontierOwners.buffer).toString("base64"),
      },
      local: {
        radius,
        size: patchSize,
        count: localTiles.length,
        planes: [...PRECISION_V3_LOCAL_PLANES],
        dtype: "uint8",
        centerTileDtype: "int32-le",
        centerTiles: Buffer.from(centers.buffer).toString("base64"),
        data: Buffer.from(local.buffer).toString("base64"),
      },
    };
  }

  private getPrecisionV3Cache(): PrecisionV3Cache {
    const game = this.requireRunner().game;
    const width = game.width();
    const height = game.height();
    // Land count changes on the impact tick; waterGraphVersion may lag until
    // the throttled navigation rebuild up to 20 ticks later.
    const terrainVersion = game.numLandTiles();
    if (
      this.precisionV3Cache !== null &&
      this.precisionV3Cache.width === width &&
      this.precisionV3Cache.height === height &&
      this.precisionV3Cache.terrainVersion === terrainVersion
    ) {
      return this.precisionV3Cache;
    }
    const size = PRECISION_V3_SIZE;
    const cellCount = size * size;
    const cellByTile = new Int32Array(width * height);
    const landDenom = new Uint32Array(cellCount);
    const staticPlanes = new Uint8Array(PRECISION_V3_PLANES.length * cellCount);
    const offset = (name: (typeof PRECISION_V3_PLANES)[number]) =>
      PRECISION_V3_PLANES.indexOf(name) * cellCount;
    game.forEachTile((tile) => {
      const cell = Math.min(size - 1, Math.floor((game.y(tile) / height) * size)) * size +
        Math.min(size - 1, Math.floor((game.x(tile) / width) * size));
      cellByTile[tile] = cell;
      if (game.isLand(tile)) {
        landDenom[cell]++;
        staticPlanes[offset("landPresence") + cell] = 255;
      } else staticPlanes[offset("waterPresence") + cell] = 255;
      if (game.terrainType(tile) === TerrainType.Highland) {
        staticPlanes[offset("highlandPresence") + cell] = 255;
      }
      if (game.terrainType(tile) === TerrainType.Mountain) {
        staticPlanes[offset("mountainPresence") + cell] = 255;
      }
      if (game.isShore(tile)) staticPlanes[offset("shorePresence") + cell] = 255;
    });
    this.precisionV3Cache = {
      width, height, terrainVersion, cellByTile, landDenom, staticPlanes,
    };
    return this.precisionV3Cache;
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
    const respondingClients = this.config.urgentEventYield
      ? [...this.urgentResponsePendingByClient]
      : [];
    for (const cid of respondingClients) this.urgentResponsePendingByClient.delete(cid);
    const maxMicroTicks = respondingClients.length > 0 ? 1 : this.config.decisionInterval;
    let ticksAdvanced = 0;
    let urgentYield = false;
    for (let i = 0; i < maxMicroTicks; i++) {
      if (this.isDone()) break;
      this.enqueueTurn(i === 0 ? stampedIntents : []);
      this.runner!.executeNextTick();
      ticksAdvanced++;
      if (this.lastError !== null) {
        throw new Error(this.lastError.errMsg);
      }
      if (this.config.urgentEventYield && respondingClients.length === 0 &&
          this.urgentResponsePendingByClient.size > 0) {
        urgentYield = true;
        break;
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
        ticksAdvanced,
        urgentYield,
        urgentResponseTick: respondingClients.length > 0,
        urgentClients: urgentYield
          ? [...this.urgentResponsePendingByClient]
          : respondingClients,
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

  stepIntentsLean(intents: Intent[]) {
    const started = performance.now();
    this.requireRunner();
    const prepared = this.prepareRawIntents(intents.slice(0, MAX_RAW_INTENTS_PER_STEP));
    const rewardAdjustment = prepared.accepted * 0.01 - prepared.rejected * 0.05;
    const before = this.previousMetrics ?? this.metrics();
    const respondingClients = this.config.urgentEventYield
      ? [...this.urgentResponsePendingByClient]
      : [];
    for (const cid of respondingClients) this.urgentResponsePendingByClient.delete(cid);
    const maxMicroTicks = respondingClients.length > 0 ? 1 : this.config.decisionInterval;
    const simStarted = performance.now();
    let ticksAdvanced = 0;
    let urgentYield = false;
    for (let i = 0; i < maxMicroTicks; i++) {
      if (this.isDone()) break;
      this.enqueueTurn(i === 0 ? prepared.stamped : []);
      this.runner!.executeNextTick();
      ticksAdvanced++;
      if (this.lastError !== null) throw new Error(this.lastError.errMsg);
      if (this.config.urgentEventYield && respondingClients.length === 0 &&
          this.urgentResponsePendingByClient.size > 0) {
        urgentYield = true;
        break;
      }
    }
    const simMs = performance.now() - simStarted;
    const after = this.metrics();
    const reward = this.reward(before, after) + rewardAdjustment;
    this.previousMetrics = after;
    this.decisionNumber++;
    const observation = this.leanObservationV3(this.clientID);
    const winner = this.winner();
    return {
      observation,
      reward,
      done: this.isDone(),
      info: {
        map: this.config.map,
        difficulty: this.config.difficulty,
        winner,
        won: isWinnerForClient(winner, this.clientID),
        turnNumber: this.turnNumber,
        tick: this.runner?.game.ticks() ?? 0,
        liveOpponents: this.liveOpponentCounts(),
        expectedOpponents: this.expectedOpponentCounts(),
        maxTurnNumber: this.maxTurnNumber,
        rawIntentAccepted: prepared.accepted,
        rawIntentRejected: prepared.rejected,
        rawIntentReasons: prepared.reasons,
        urgentYield,
        urgentResponseTick: respondingClients.length > 0,
        urgentClients: urgentYield
          ? [...this.urgentResponsePendingByClient]
          : respondingClients,
        timings: {
          simMs,
          ticksAdvanced,
          totalMs: performance.now() - started,
        },
      },
    };
  }

  stepReplayTakeoverLean(
    intents: Intent[],
    recordedTurns: StampedIntent[][],
  ) {
    const started = performance.now();
    this.requireRunner();
    const prepared = this.prepareRawIntents(intents.slice(0, MAX_RAW_INTENTS_PER_STEP));
    const rewardAdjustment = prepared.accepted * 0.01 - prepared.rejected * 0.05;
    const before = this.previousMetrics ?? this.metrics();
    const respondingClients = this.config.urgentEventYield
      ? [...this.urgentResponsePendingByClient]
      : [];
    for (const cid of respondingClients) this.urgentResponsePendingByClient.delete(cid);
    const maxMicroTicks = respondingClients.length > 0 ? 1 : this.config.decisionInterval;
    const simStarted = performance.now();
    let ticksAdvanced = 0;
    let opponentIntentsApplied = 0;
    let urgentYield = false;
    for (let i = 0; i < maxMicroTicks; i++) {
      if (this.isDone()) break;
      const opponents = (recordedTurns[i] ?? []).filter(
        (intent) => intent.clientID !== this.clientID,
      );
      opponentIntentsApplied += opponents.length;
      this.enqueueTurn([
        ...(i === 0 ? prepared.stamped : []),
        ...opponents,
      ]);
      this.runner!.executeNextTick();
      ticksAdvanced++;
      if (this.lastError !== null) throw new Error(this.lastError.errMsg);
      if (this.config.urgentEventYield && respondingClients.length === 0 &&
          this.urgentResponsePendingByClient.size > 0) {
        urgentYield = true;
        break;
      }
    }
    const simMs = performance.now() - simStarted;
    const after = this.metrics();
    const reward = this.reward(before, after) + rewardAdjustment;
    this.previousMetrics = after;
    this.decisionNumber++;
    const observation = this.leanObservationV3(this.clientID);
    const winner = this.winner();
    return {
      observation,
      reward,
      done: this.isDone(),
      info: {
        map: this.config.map,
        difficulty: this.config.difficulty,
        winner,
        won: isWinnerForClient(winner, this.clientID),
        turnNumber: this.turnNumber,
        tick: this.runner?.game.ticks() ?? 0,
        liveOpponents: this.liveOpponentCounts(),
        expectedOpponents: this.expectedOpponentCounts(),
        maxTurnNumber: this.maxTurnNumber,
        rawIntentAccepted: prepared.accepted,
        rawIntentRejected: prepared.rejected,
        rawIntentReasons: prepared.reasons,
        opponentIntentsApplied,
        urgentYield,
        urgentResponseTick: respondingClients.length > 0,
        urgentClients: urgentYield
          ? [...this.urgentResponsePendingByClient]
          : respondingClients,
        timings: {
          simMs,
          ticksAdvanced,
          totalMs: performance.now() - started,
        },
      },
    };
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
      const requestedTroops = Math.floor(intent.troops ?? 1);
      if (!Number.isFinite(requestedTroops) || requestedTroops < 1 ||
          requestedTroops > Math.floor(player.troops())) {
        return { reason: "attack:invalid_troops" };
      }
      if (intent.targetID === null) {
        if (!hasLandBorderWithTerraNullius(game, player)) {
          return { reason: "attack:terra_not_bordering" };
        }
        return { intent: { ...intent, troops: requestedTroops } };
      }
      if (!game.hasPlayer(intent.targetID)) return { reason: "attack:missing_target" };
      const target = game.player(intent.targetID);
      if (!target.isAlive()) return { reason: "attack:target_dead" };
      if (!player.sharesBorderWith(target) || !player.canAttackPlayer(target, true)) {
        return { reason: "attack:cannot_attack_target" };
      }
      return { intent: { ...intent, troops: requestedTroops } };
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
      const troops = Math.floor(intent.troops);
      if (!Number.isFinite(troops) || troops < 1 || troops > Math.floor(player.troops())) {
        return { reason: "boat:invalid_troops" };
      }
      return { intent: { ...intent, troops } };
    }
    if (intent.type === "build_unit") {
      if (!PlayerBuildable.has(intent.unit)) return { reason: "build:invalid_unit" };
      if (!game.isValidRef(intent.tile)) return { reason: "build:invalid_tile" };
      if (!this.canBuildUnitType(player, intent.unit)) {
        return { reason: `build:${intent.unit}:unaffordable_or_disabled` };
      }
      const spawnTile = player.canBuild(intent.unit, intent.tile);
      if (spawnTile === false) return { reason: `build:${intent.unit}:cannot_build_here` };
      if (
        !TARGET_PRESERVING_BUILD_UNITS.has(intent.unit) &&
        spawnTile !== intent.tile
      ) {
        return { reason: `build:${intent.unit}:stale_exact_tile` };
      }
      return {
        intent: {
          ...intent,
          // Nukes and warships return a SOURCE silo/port from canBuild.
          // ConstructionExecution needs the requested TARGET and independently
          // resolves that source again.
          tile: intent.tile,
        },
      };
    }
    if (intent.type === "cancel_attack") {
      if (!player.outgoingAttacks().some(
        (attack) => attack.id() === intent.attackID && !attack.retreating(),
      )) {
        return { reason: "cancel_attack:missing_attack" };
      }
      return { intent };
    }
    if (intent.type === "upgrade_structure") {
      const unit = game.unit(intent.unitId);
      if (
        unit === undefined ||
        unit.owner() !== player ||
        unit.type() !== intent.unit ||
        !player.canUpgradeUnit(unit)
      ) {
        return { reason: "upgrade_structure:missing_or_unavailable_unit" };
      }
      return { intent };
    }
    if (intent.type === "delete_unit") {
      const unit = game.unit(intent.unitId);
      if (
        unit === undefined ||
        unit.owner() !== player ||
        !unit.isActive() ||
        unit.isMarkedForDeletion() ||
        !game.isLand(unit.tile()) ||
        game.owner(unit.tile()) !== player ||
        !player.canDeleteUnit()
      ) {
        return { reason: "delete_unit:missing_or_unavailable_unit" };
      }
      return { intent };
    }
    if (intent.type === "cancel_boat") {
      if (!player.units(UnitType.TransportShip).some(
        (unit) => unit.id() === intent.unitID &&
          !unit.transportShipState().isRetreating,
      )) {
        return { reason: "cancel_boat:missing_boat" };
      }
      return { intent };
    }
    if (intent.type === "move_warship") {
      if (!game.isValidRef(intent.tile) || !game.isWater(intent.tile)) {
        return { reason: "move_warship:invalid_water_tile" };
      }
      const ownedWarships = new Set(
        player.units(UnitType.Warship).filter((unit) => unit.isActive())
          .map((unit) => unit.id()),
      );
      if (intent.unitIds.length === 0 || !intent.unitIds.every((unitID) => ownedWarships.has(unitID))) {
        return { reason: "move_warship:missing_warship" };
      }
      if (intent.unitIds.some((unitID) => {
        const unit = game.unit(unitID)!;
        return intent.tile === unit.tile() ||
          intent.tile === unit.warshipState().patrolTile;
      })) return { reason: "move_warship:stale_or_current_target" };
      const targetComponent = game.getWaterComponent(intent.tile);
      if (targetComponent === null || !intent.unitIds.every((unitID) => {
        const unit = game.unit(unitID)!;
        return game.getWaterComponent(unit.tile()) === targetComponent;
      })) return { reason: "move_warship:unreachable_component" };
      return { intent };
    }
    if (intent.type === "allianceRequest") {
      if (!game.hasPlayer(intent.recipient) ||
          !player.canSendAllianceRequest(game.player(intent.recipient))) {
        return { reason: "allianceRequest:unavailable_recipient" };
      }
      return { intent };
    }
    if (intent.type === "allianceExtension") {
      if (!game.hasPlayer(intent.recipient)) return { reason: "allianceExtension:missing_recipient" };
      if (player.allianceInfo(game.player(intent.recipient))?.canExtend !== true) {
        return { reason: "allianceExtension:unavailable" };
      }
      return { intent };
    }
    if (intent.type === "allianceReject") {
      if (!player.incomingAllianceRequests().some(
        (request) => request.requestor().id() === intent.requestor,
      )) return { reason: "allianceReject:missing_request" };
      return { intent };
    }
    if (intent.type === "breakAlliance") {
      if (!game.hasPlayer(intent.recipient) ||
          player.allianceWith(game.player(intent.recipient)) === null) {
        return { reason: "breakAlliance:missing_alliance" };
      }
      return { intent };
    }
    if (intent.type === "targetPlayer") {
      if (!game.hasPlayer(intent.target) || !player.canTarget(game.player(intent.target))) {
        return { reason: "targetPlayer:unavailable_target" };
      }
      return { intent };
    }
    if (intent.type === "embargo") {
      if (intent.action !== "start" && intent.action !== "stop") {
        return { reason: "embargo:invalid_action" };
      }
      if (!game.hasPlayer(intent.targetID) || game.player(intent.targetID) === player) {
        return { reason: "embargo:missing_target" };
      }
      const target = game.player(intent.targetID);
      const active = player.hasEmbargoAgainst(target);
      if ((intent.action === "start" && active) || (intent.action === "stop" && !active)) {
        return { reason: "embargo:stale_action" };
      }
      return { intent };
    }
    if (intent.type === "donate_gold") {
      if (!game.hasPlayer(intent.recipient)) return { reason: "donate_gold:missing_recipient" };
      const recipient = game.player(intent.recipient);
      if (!player.canDonateGold(recipient)) return { reason: "donate_gold:unavailable" };
      const max = Math.max(0, Number(player.gold()));
      const requested = intent.gold ?? Math.floor(max / 3);
      if (!Number.isFinite(requested) || max < 1 || requested < 1 || requested > max) {
        return { reason: "donate_gold:invalid_amount" };
      }
      return { intent: { ...intent, gold: Math.floor(requested) } };
    }
    if (intent.type === "donate_troops") {
      if (!game.hasPlayer(intent.recipient)) return { reason: "donate_troops:missing_recipient" };
      const recipient = game.player(intent.recipient);
      if (!player.canDonateTroops(recipient)) return { reason: "donate_troops:unavailable" };
      const max = Math.max(0, Math.floor(player.troops()));
      const requested = intent.troops ?? game.config().defaultDonationAmount(player);
      if (!Number.isFinite(requested) || max < 1 || requested < 1 || requested > max) {
        return { reason: "donate_troops:invalid_amount" };
      }
      return { intent: { ...intent, troops: Math.floor(requested) } };
    }
    if (intent.type === "emoji") {
      const recipient = intent.recipient === AllPlayers
        ? AllPlayers
        : game.hasPlayer(intent.recipient) ? game.player(intent.recipient) : null;
      if (recipient === null || recipient === player || !player.canSendEmoji(recipient)) {
        return { reason: "emoji:unavailable_recipient" };
      }
      if (!(EMOJIS as readonly number[]).includes(intent.emoji)) {
        return { reason: "emoji:unsupported_value" };
      }
      return { intent };
    }
    if (intent.type === "quick_chat") {
      if (!game.hasPlayer(intent.recipient)) {
        return { reason: "quick_chat:missing_recipient" };
      }
      const recipient = game.player(intent.recipient);
      if (recipient === player || !player.canSendQuickChat(recipient)) {
        return { reason: "quick_chat:unavailable_recipient" };
      }
      const entry = QUICK_CHAT_KEYS.find((option) => option.key === intent.quickChatKey);
      if (entry === undefined) return { reason: "quick_chat:unsupported_key" };
      if (entry.requiresPlayer &&
          (intent.target === undefined || !game.hasPlayer(intent.target))) {
        return { reason: "quick_chat:missing_target" };
      }
      return { intent };
    }
    if (intent.type === "embargo_all") {
      if ((intent.action !== "start" && intent.action !== "stop") ||
          !player.canEmbargoAll()) {
        return { reason: "embargo_all:unavailable" };
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
    const respondingClients = this.config.urgentEventYield
      ? [...this.urgentResponsePendingByClient]
      : [];
    for (const cid of respondingClients) this.urgentResponsePendingByClient.delete(cid);
    const maxMicroTicks = respondingClients.length > 0 ? 1 : this.config.decisionInterval;
    let ticksAdvanced = 0;
    let urgentYield = false;
    for (let i = 0; i < maxMicroTicks; i++) {
      if (this.isDone()) break;
      this.enqueueTurn(i === 0 ? stampedIntents : []);
      this.runner!.executeNextTick();
      ticksAdvanced++;
      if (this.lastError !== null) {
        throw new Error(this.lastError.errMsg);
      }
      if (this.config.urgentEventYield && respondingClients.length === 0 &&
          this.urgentResponsePendingByClient.size > 0) {
        urgentYield = true;
        break;
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
        ticksAdvanced,
      },
      started,
      {
        ...extraInfo,
        urgentYield,
        urgentResponseTick: respondingClients.length > 0,
        urgentClients: urgentYield
          ? [...this.urgentResponsePendingByClient]
          : respondingClients,
      },
    );
  }

  transcript(): Array<{ turn: number; candidate: ActionCandidate }> {
    return this.actionTranscript;
  }

  executionTranscript(): ExecutionTranscriptEntry[] {
    return this.nativeExecutionTranscript;
  }

  private captureExecution(tick: number, execution: Execution): void {
    if (
      this.nativeExecutionTranscript.length >= MAX_EXECUTION_TRANSCRIPT_ENTRIES
    ) {
      return;
    }
    const fields: ExecutionTranscriptEntry["fields"] = {};
    for (const [name, value] of Object.entries(
      execution as unknown as Record<string, unknown>,
    )) {
      if (
        value === null ||
        typeof value === "string" ||
        typeof value === "number" ||
        typeof value === "boolean"
      ) {
        fields[name] = value;
        continue;
      }
      if (typeof value !== "object") continue;
      const player = value as Partial<Player>;
      if (typeof player.isPlayer !== "function" || !player.isPlayer()) continue;
      fields[name] = {
        id: player.id!(),
        smallID: player.smallID!(),
        type: player.type!(),
        clientID: player.clientID!(),
        displayName: player.displayName!(),
      };
    }
    this.nativeExecutionTranscript.push({
      tick,
      type: execution.constructor.name,
      fields,
    });
  }

  gameRecord(): GameRecord {
    const runner = this.requireRunner();
    if (this.gameStartInfo === null) {
      throw new Error("environment has not been reset");
    }
    const allStats = runner.game.stats().stats();
    const playerRecords: PlayerRecord[] = this.controlledClientIDs.map((clientID) => ({
      clientID,
      username: clientID === this.clientID ? USERNAME : clientID,
      clanTag: null,
      persistentID: null,
      stats: allStats[clientID],
    }));
    const end = this.startedAt + this.turnNumber * 100;
    const partial = createPartialGameRecord(
      this.gameStartInfo.gameID,
      this.gameStartInfo.config,
      playerRecords,
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
    const players = this.policyPlayers(cid)
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
          (humanVisibleNumberFloor(metrics.troops / 10) * 10) / 10_000_000,
          humanVisibleNumberFloor(metrics.gold) / 10_000_000,
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
    this.legacyCandidateGenerationCount++;
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
        // Nukes and warships return a source silo/port from canBuild, not the
        // requested target. Preserve the probed target in the wire intent.
        this.addBuildIntentCandidate(
          add,
          game,
          seen,
          unitType,
          TARGET_PRESERVING_BUILD_UNITS.has(unitType) ? tile : canBuild,
        );
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

  private tileBitset(tiles: readonly TileRef[]) {
    const game = this.requireRunner().game;
    const tileCount = game.width() * game.height();
    const bits = new Uint8Array(Math.ceil(tileCount / 8));
    for (const tile of tiles) bits[tile >> 3] |= 1 << (tile & 7);
    return {
      count: tiles.length,
      tileCount,
      dtype: "bitset-deflate-raw" as const,
      data: tiles.length === 0
        ? ""
        : Buffer.from(deflateRawSync(bits, { level: 1 })).toString("base64"),
    };
  }

  private waterTilesByComponent(): Map<number, TileRef[]> {
    const game = this.requireRunner().game;
    const version = game.waterGraphVersion();
    if (this.waterTilesByComponentCache !== null &&
        this.waterComponentCacheVersion === version) {
      return this.waterTilesByComponentCache;
    }
    const components = new Map<number, TileRef[]>();
    game.forEachTile((tile) => {
      if (!game.isWater(tile)) return;
      const component = game.getWaterComponent(tile);
      if (component === null) return;
      const tiles = components.get(component) ?? [];
      tiles.push(tile);
      components.set(component, tiles);
    });
    this.waterTilesByComponentCache = components;
    this.waterComponentCacheVersion = version;
    this.waterComponentBitsetsCache = new Map();
    return components;
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
    const game = this.requireRunner().game;
    // Immediate terrain signature for land-to-water nuclear conversion.
    const version = game.numLandTiles();
    if (this.shoreLandTilesCache !== null &&
        this.shoreLandTilesCacheVersion === version) {
      return this.shoreLandTilesCache;
    }
    const tiles: TileRef[] = [];
    game.forEachTile((tile) => {
      if (game.isLand(tile) && game.isShore(tile)) tiles.push(tile);
    });
    this.shoreLandTilesCache = tiles;
    this.shoreLandTilesCacheVersion = version;
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

  private placementBuildOptions(
    player: Player,
  ): { unit: string; tile: number; rocketDirectionUp: boolean }[] {
    const defaultLimit = this.config.actionProfile === "bot-lite" ? 24 : 80;
    const probeLimit = this.config.actionProfile === "bot-lite"
      ? defaultLimit
      : this.config.buildProbeLimit;
    const probes = this.buildProbeTiles(player);
    const tiles = probeLimit === "full"
      ? probes
      : sampleEvenly(probes, probeLimit || defaultLimit);
    const unitTypes: PlayerBuildableUnitType[] = [
      ...LAND_STRUCTURE_BUILD_UNITS,
      UnitType.Port,
    ];
    const options: {
      unit: string;
      tile: number;
      rocketDirectionUp: boolean;
    }[] = [];
    const seen = new Set<string>();
    for (const tile of tiles) {
      const validTiles = this.validStructureSpawnTiles(player, tile);
      for (const unitType of unitTypes) {
        if (!this.canBuildUnitType(player, unitType)) continue;
        const placement = player.canBuild(unitType, tile, validTiles);
        if (placement === false) continue;
        const key = `${unitType}:${placement}`;
        if (seen.has(key)) continue;
        seen.add(key);
        options.push({ unit: unitType, tile: placement, rocketDirectionUp: true });
      }
    }
    return options;
  }

  private placementBuildProposalOptions(
    player: Player,
  ): { unit: string; tile: number; rocketDirectionUp: boolean }[] {
    const unitTypes: PlayerBuildableUnitType[] = [
      ...LAND_STRUCTURE_BUILD_UNITS,
      UnitType.Port,
    ];
    const options: { unit: string; tile: number; rocketDirectionUp: boolean }[] = [];
    const seen = new Set<string>();
    const game = this.requireRunner().game;
    const anchors = new Set<TileRef>();
    const priority = new Set<TileRef>();
    for (const tile of player.borderTiles()) {
      anchors.add(tile);
      for (const neighbor of game.neighbors(tile)) anchors.add(neighbor);
    }
    for (const unit of player.units()) {
      anchors.add(unit.tile());
      if (unit.type() === UnitType.Port && unit.isActive() && !unit.isUnderConstruction()) {
        for (const neighbor of game.neighbors(unit.tile())) {
          if (game.isWater(neighbor)) priority.add(neighbor);
        }
      }
    }
    const configuredLimit = this.config.buildProbeLimit;
    const limit = configuredLimit === "full"
      ? priority.size + anchors.size
      : configuredLimit;
    const priorityTiles = [...priority].sort((a, b) => a - b)
      .slice(0, limit);
    const remaining = [...anchors]
      .filter((tile) => game.isValidRef(tile) && !priority.has(tile))
      .sort((a, b) => a - b);
    const tiles = [
      ...priorityTiles,
      ...sampleEvenly(remaining, Math.max(0, limit - priorityTiles.length)),
    ];
    for (const tile of tiles) {
      const validTiles = this.validStructureSpawnTiles(player, tile);
      for (const unit of unitTypes) {
        if (!this.canBuildUnitType(player, unit)) continue;
        const placement = player.canBuild(unit, tile, validTiles);
        if (placement === false) continue;
        const key = `${unit}:${placement}`;
        if (seen.has(key)) continue;
        seen.add(key);
        options.push({ unit, tile: placement as number, rocketDirectionUp: true });
      }
    }
    if (this.canBuildUnitType(player, UnitType.Warship)) {
      for (const tile of tiles) {
        if (!game.isWater(tile) || player.canBuild(UnitType.Warship, tile) === false) {
          continue;
        }
        const key = `${UnitType.Warship}:${tile}`;
        if (seen.has(key)) continue;
        seen.add(key);
        options.push({ unit: UnitType.Warship, tile, rocketDirectionUp: true });
      }
    }
    return options;
  }

  private visibleEntityTokens(cid: ClientID) {
    const game = this.requireRunner().game;
    const me = game.playerByClientID(cid);
    const width = Math.max(1, game.width());
    const height = Math.max(1, game.height());
    const tokenTypes = [...PRECISION_V3_ENTITY_KINDS];
    const kindIndex = new Map(tokenTypes.map((type, index) => [type, index]));
    return game.units(...tokenTypes)
      .filter((unit) => unit.isActive() || unit.isUnderConstruction())
      .map((unit) => {
        const tile = unit.tile();
        const owner = unit.owner();
        return {
          kind: kindIndex.get(
            unit.type() as (typeof PRECISION_V3_ENTITY_KINDS)[number],
          ) ?? 0,
          owner: owner.smallID(),
          rel: owner === me ? 0 : owner.isPlayer() ? 2 : 1,
          x: game.x(tile) / width,
          y: game.y(tile) / height,
          troops: Math.log1p(
            humanVisibleNumberFloor(Math.max(0, unit.troops?.() ?? 0) / 10) * 10,
          ),
          health: Math.log1p(Math.max(0, Number(unit.health?.() ?? 0))),
          tile: tile as number,
          id: unit.id(),
          unitType: unit.type(),
          level: unit.level(),
          active: unit.isActive(),
          underConstruction: unit.isUnderConstruction(),
          targetTile: TARGET_PRESERVING_BUILD_UNITS.has(unit.type())
            ? (unit.targetTile() ?? null)
            : null,
          markedForDeletion: unit.isMarkedForDeletion(),
          trainType: unit.trainType() ?? null,
          loaded: unit.isLoaded() ?? null,
          hasTrainStation: unit.hasTrainStation(),
        };
      });
  }

  private missileTargetTiles(
    player: Player,
  ): { tile: TileRef; targetPlayerID: string }[] {
    const game = this.requireRunner().game;
    const byTile = new Map<TileRef, string>();
    for (const unit of game.units()) {
      const owner = unit.owner();
      if (!owner.isPlayer() || owner === player || !owner.isAlive()) continue;
      byTile.set(unit.tile(), owner.id());
    }
    for (const target of game.players()) {
      if (target === player || !target.isAlive() || !target.hasSpawned()) continue;
      for (const tile of sampleEvenly([...target.tiles()], 16)) {
        byTile.set(tile, target.id());
      }
    }
    return [...byTile]
      .map(([tile, targetPlayerID]) => ({ tile, targetPlayerID }))
      .sort((a, b) =>
        a.targetPlayerID.localeCompare(b.targetPlayerID) || a.tile - b.tile,
      );
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
    "bots" | "difficulty" | "map" | "nations" | "seed" | "infiniteGold" |
    "donateGold" | "donateTroops" | "waterNukes"
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
      infiniteGold: config.infiniteGold ?? false,
      infiniteTroops: false,
      instantBuild: false,
      randomSpawn: false,
      donateGold: config.donateGold,
      donateTroops: config.donateTroops,
      waterNukes: config.waterNukes ? true : undefined,
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
  executionObserver?: (tick: number, execution: Execution) => void,
  nativeTeacherClientID?: ClientID,
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
  if (executionObserver !== undefined) {
    const addExecution = game.addExecution.bind(game);
    game.addExecution = (...executions: Execution[]) => {
      for (const execution of executions) {
        executionObserver(game.ticks(), execution);
      }
      addExecution(...executions);
    };
  }
  const runner = new GameRunner(
    game,
    new Executor(game, gameStart.gameID, clientID),
    callBack,
  );
  runner.init();
  if (nativeTeacherClientID !== undefined) {
    const teacher = game.playerByClientID(nativeTeacherClientID);
    if (teacher === null) {
      throw new Error(`native teacher client ${nativeTeacherClientID} not found`);
    }
    game.addExecution(
      new NationExecution(gameStart.gameID, new Nation(undefined, teacher.info())),
    );
  }
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

export function humanVisibleNumberFloor(value: number): number {
  const n = Math.max(0, value);
  if (n >= 10_000_000) return Math.floor(n / 100_000) * 100_000;
  if (n >= 1_000_000) return Math.floor(n / 10_000) * 10_000;
  if (n >= 100_000) return Math.floor(n / 1_000) * 1_000;
  if (n >= 10_000) return Math.floor(n / 100) * 100;
  if (n >= 1_000) return Math.floor(n / 10) * 10;
  return Math.floor(n);
}

function summarizePlayer(
  player: Player,
  selfID: PlayerID | null,
): PlayerSummary {
  const units: Partial<Record<UnitType, number>> = {};
  for (const unit of player.units()) {
    units[unit.type()] = (units[unit.type()] ?? 0) + 1;
  }
  const isSelf = player.id() === selfID;
  const troops = player.troops();
  const gold = Number(player.gold());
  return {
    id: player.id(),
    smallID: player.smallID(),
    type: player.type(),
    isSelf,
    isAlive: player.isAlive(),
    tiles: player.numTilesOwned(),
    // The client displays opponent gold via renderNumber and troops via
    // renderTroops (renderNumber(troops / 10)). Keep the information but not
    // hidden precision below those display buckets. The same display contract
    // applies to self and opponents.
    troops: humanVisibleNumberFloor(troops / 10) * 10,
    gold: humanVisibleNumberFloor(gold),
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
