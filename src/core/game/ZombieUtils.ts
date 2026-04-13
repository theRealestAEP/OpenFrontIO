import { GameConfig } from "../Schemas";
import {
  CureState,
  Game,
  Player,
  PlayerInfo,
  PlayerSpecialRole,
  PlayerType,
  Team,
} from "./Game";

type SpecialRoleMethodCarrier = {
  specialRole(): PlayerSpecialRole | null;
};

type SpecialRoleFieldCarrier = {
  specialRole: PlayerSpecialRole | null;
};

type CureStateCarrier = {
  cureState(): CureState;
};

export const ZOMBIE_TERRITORY_COLOR = "rgb(118,20,20)";
export const ZOMBIE_TRANSPORT_DEATH_BURST_DAMAGE = 100;
export const ZOMBIE_START_DELAY_TICKS = 600;
export const ZOMBIE_CURE_RESEARCH_TICKS = 6000;
export const ZOMBIE_MAX_TROOPS_MULTIPLIER = 8;
export const ZOMBIE_TROOP_REGEN_MULTIPLIER = 36;
export const ZOMBIE_INFESTATION_TILES_PER_CAP_BONUS = 180;
export const ZOMBIE_INFESTATION_CAP_BONUS = 150_000;
export const ZOMBIE_BOAT_CAP = 30;
export const ZOMBIE_ATTACK_RATE_MIN = 3;
export const ZOMBIE_ATTACK_RATE_MAX = 5;
export const ZOMBIE_MAX_ACTIVE_LAND_ATTACKS = 8;
export const ZOMBIE_LATE_GAME_MAX_ACTIVE_LAND_ATTACKS = 24;
export const ZOMBIE_MAX_ACTIVE_ATTACKS_PER_TARGET = 3;
export const ZOMBIE_MAX_FOCUSED_TARGETS = 2;
export const ZOMBIE_MAX_TOTAL_PRESSURE_OPERATIONS = 40;
export const ZOMBIE_LATE_GAME_MAX_TOTAL_PRESSURE_OPERATIONS = 48;
export const ZOMBIE_FRONT_SCAN_LIMIT = 640;
export const ZOMBIE_FRONTS_PER_TARGET = 5;
export const ZOMBIE_LAND_ATTACKS_PER_CYCLE = 4;
export const ZOMBIE_LATE_GAME_LAND_ATTACKS_PER_CYCLE = 10;
export const ZOMBIE_MAX_ACTIVE_TRANSPORTS = 10;
export const ZOMBIE_LATE_GAME_MAX_ACTIVE_TRANSPORTS = 18;
export const ZOMBIE_WATER_ATTACKS_PER_CYCLE = 2;
export const ZOMBIE_LATE_GAME_WATER_ATTACKS_PER_CYCLE = 5;
export const ZOMBIE_TARGET_SHORE_SCAN_LIMIT = 8;
export const ZOMBIE_OPEN_LAND_ATTACKS_PER_CYCLE = 1;
export const ZOMBIE_LAND_ATTACK_TROOP_SHARE = 0.22;
export const ZOMBIE_WATER_ATTACK_TROOP_SHARE = 0.12;
export const ZOMBIE_MAX_LAND_ATTACK_TROOPS = 400_000;
export const ZOMBIE_MAX_WATER_ATTACK_TROOPS = 250_000;
export const ZOMBIE_LATE_GAME_MAX_LAND_ATTACK_TROOPS = 1_500_000;
export const ZOMBIE_LATE_GAME_MAX_WATER_ATTACK_TROOPS = 800_000;
export const ZOMBIE_MAX_STARTUP_RETALIATION_TROOPS = 500_000;
export const ZOMBIE_MIN_LAND_ATTACK_TROOPS = 30_000;
export const ZOMBIE_MIN_WATER_ATTACK_TROOPS = 20_000;
export const ZOMBIE_LAND_ATTACK_TARGET_MAX_TROOPS_RATIO = 0.2;
export const ZOMBIE_WATER_ATTACK_TARGET_MAX_TROOPS_RATIO = 0.14;
export const ZOMBIE_ENDGAME_WAVE_RAMP_TICKS = 21_000;
export const ZOMBIE_WAVE_TIME_SCALE_TICKS = 6000;
export const ZOMBIE_WAVE_TIME_SCALE_MAX = 5;
export const ZOMBIE_BEACHHEAD_WAVE_BONUS = 1.25;
export const ZOMBIE_RESERVE_RATIO = 0.01;
export const ZOMBIE_TRIGGER_RATIO = 0.03;
export const ZOMBIE_EXPAND_RATIO = 0.55;
export const ZOMBIE_NAVAL_REINFORCEMENT_FRONT_LIMIT = 4;
export const ZOMBIE_NAVAL_TARGET_MEMORY_TICKS = 600;
export const ZOMBIE_FOCUSED_TARGET_RAMP_TICKS = 8400;
export const ZOMBIE_LATE_GAME_MAX_FOCUSED_TARGETS = 24;
export const ZOMBIE_STARTUP_RETALIATION_ATTACKS_PER_CYCLE = 3;
export const ZOMBIE_STARTUP_RETALIATION_TROOP_SHARE = 0.4;
export const ZOMBIE_UNCURED_ATTACK_INTERVAL_TICKS = 8;
export const ZOMBIE_UNCURED_ATTACKER_LOSS_MULTIPLIER = 12;
export const ZOMBIE_UNCURED_DEFENDER_LOSS_MULTIPLIER = 0.025;
export const ZOMBIE_UNCURED_DEFENSE_FLOOR_PER_TILE = 900;
export const ZOMBIE_MAX_ATTACK_STEPS_PER_TICK = 800;
export const ZOMBIE_MAX_TERRA_NULLIUS_ATTACK_STEPS_PER_TICK = 300;

function resolveSpecialRole(
  value: SpecialRoleMethodCarrier | SpecialRoleFieldCarrier,
): PlayerSpecialRole | null {
  if ("specialRole" in value && typeof value.specialRole === "function") {
    return value.specialRole();
  }
  return value.specialRole;
}

export function isZombieSpecialRole(
  specialRole: PlayerSpecialRole | null | undefined,
): boolean {
  return specialRole === PlayerSpecialRole.Zombie;
}

export function isZombiePlayer(
  player: Player | SpecialRoleMethodCarrier | SpecialRoleFieldCarrier,
): boolean {
  return isZombieSpecialRole(resolveSpecialRole(player));
}

export function isZombiePlayerInfo(playerInfo: PlayerInfo): boolean {
  return isZombieSpecialRole(playerInfo.specialRole);
}

export function isZombieRulesetConfig(
  gameConfig: Pick<GameConfig, "specialRuleset">,
): boolean {
  return gameConfig.specialRuleset === "zombie_survival";
}

export function isZombieRulesetGame(game: Pick<Game, "config">): boolean {
  return isZombieRulesetConfig(game.config().gameConfig());
}

export function hasZombieStartDelayProtection(
  game: Pick<Game, "config" | "ticks">,
  player: Player | SpecialRoleMethodCarrier | SpecialRoleFieldCarrier,
): boolean {
  return (
    isZombieRulesetGame(game) &&
    isZombiePlayer(player) &&
    game.ticks() < ZOMBIE_START_DELAY_TICKS
  );
}

export function isCuredPlayer(player: CureStateCarrier): boolean {
  return player.cureState() === CureState.Cured;
}

export function canAutoAnnexTarget(
  attacker: CureStateCarrier,
  defender: Player,
): boolean {
  return !isZombiePlayer(defender) || isCuredPlayer(attacker);
}

export function canPlayerResearchCure(player: Player): boolean {
  return !isZombiePlayer(player) && player.type() !== PlayerType.Bot;
}

export function zombieResearchGroupKey(player: Player): string {
  const team = player.team();
  return team !== null ? `team:${team}` : `player:${player.id()}`;
}

export function zombieResearchTeam(team: Team | null): string | null {
  return team === null ? null : `team:${team}`;
}
