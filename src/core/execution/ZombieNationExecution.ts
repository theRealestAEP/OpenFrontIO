import {
  Execution,
  Game,
  Nation,
  Player,
  TerrainType,
  UnitType,
} from "../game/Game";
import { TileRef } from "../game/GameMap";
import { bestShoreDeploymentSource } from "../game/TransportShipUtils";
import {
  ZOMBIE_ATTACK_RATE_MAX,
  ZOMBIE_ATTACK_RATE_MIN,
  ZOMBIE_BEACHHEAD_WAVE_BONUS,
  ZOMBIE_ENDGAME_WAVE_RAMP_TICKS,
  ZOMBIE_EXPAND_RATIO,
  ZOMBIE_FOCUSED_TARGET_RAMP_TICKS,
  ZOMBIE_FRONTS_PER_TARGET,
  ZOMBIE_FRONT_SCAN_LIMIT,
  ZOMBIE_LATE_GAME_LAND_ATTACKS_PER_CYCLE,
  ZOMBIE_LATE_GAME_MAX_ACTIVE_LAND_ATTACKS,
  ZOMBIE_LATE_GAME_MAX_ACTIVE_TRANSPORTS,
  ZOMBIE_LATE_GAME_MAX_LAND_ATTACK_TROOPS,
  ZOMBIE_LATE_GAME_MAX_TOTAL_PRESSURE_OPERATIONS,
  ZOMBIE_LATE_GAME_MAX_WATER_ATTACK_TROOPS,
  ZOMBIE_LAND_ATTACKS_PER_CYCLE,
  ZOMBIE_LAND_ATTACK_TARGET_MAX_TROOPS_RATIO,
  ZOMBIE_LAND_ATTACK_TROOP_SHARE,
  ZOMBIE_LATE_GAME_MAX_FOCUSED_TARGETS,
  ZOMBIE_LATE_GAME_WATER_ATTACKS_PER_CYCLE,
  ZOMBIE_MAX_ACTIVE_ATTACKS_PER_TARGET,
  ZOMBIE_MAX_ACTIVE_LAND_ATTACKS,
  ZOMBIE_MAX_ACTIVE_TRANSPORTS,
  ZOMBIE_MAX_FOCUSED_TARGETS,
  ZOMBIE_MAX_LAND_ATTACK_TROOPS,
  ZOMBIE_MAX_TOTAL_PRESSURE_OPERATIONS,
  ZOMBIE_MAX_STARTUP_RETALIATION_TROOPS,
  ZOMBIE_MAX_WATER_ATTACK_TROOPS,
  ZOMBIE_MIN_LAND_ATTACK_TROOPS,
  ZOMBIE_MIN_WATER_ATTACK_TROOPS,
  ZOMBIE_NAVAL_REINFORCEMENT_FRONT_LIMIT,
  ZOMBIE_NAVAL_TARGET_MEMORY_TICKS,
  ZOMBIE_OPEN_LAND_ATTACKS_PER_CYCLE,
  ZOMBIE_RESERVE_RATIO,
  ZOMBIE_STARTUP_RETALIATION_ATTACKS_PER_CYCLE,
  ZOMBIE_STARTUP_RETALIATION_TROOP_SHARE,
  ZOMBIE_START_DELAY_TICKS,
  ZOMBIE_TARGET_SHORE_SCAN_LIMIT,
  ZOMBIE_TRIGGER_RATIO,
  ZOMBIE_WATER_ATTACKS_PER_CYCLE,
  ZOMBIE_WATER_ATTACK_TARGET_MAX_TROOPS_RATIO,
  ZOMBIE_WATER_ATTACK_TROOP_SHARE,
  ZOMBIE_WAVE_TIME_SCALE_MAX,
  ZOMBIE_WAVE_TIME_SCALE_TICKS,
} from "../game/ZombieUtils";
import { PseudoRandom } from "../PseudoRandom";
import { GameID } from "../Schemas";
import { simpleHash } from "../Util";
import { AttackExecution } from "./AttackExecution";
import { SpawnExecution } from "./SpawnExecution";
import { TransportShipExecution } from "./TransportShipExecution";

export class ZombieNationExecution implements Execution {
  private active = true;
  private random: PseudoRandom;
  private mg!: Game;
  private player: Player | null = null;
  private attackRate = ZOMBIE_ATTACK_RATE_MIN;
  private attackTick = 0;
  private recentNavalTargets = new Map<string, number>();

  constructor(
    private readonly gameID: GameID,
    private readonly nation: Nation,
  ) {
    this.random = new PseudoRandom(
      simpleHash(nation.playerInfo.id) + simpleHash(gameID),
    );
  }

  init(mg: Game): void {
    this.mg = mg;
    this.attackRate = this.random.nextInt(
      ZOMBIE_ATTACK_RATE_MIN,
      ZOMBIE_ATTACK_RATE_MAX + 1,
    );
    this.attackTick = this.random.nextInt(0, this.attackRate);

    if (!this.mg.hasPlayer(this.nation.playerInfo.id)) {
      this.player = this.mg.addPlayer(this.nation.playerInfo);
    } else {
      this.player = this.mg.player(this.nation.playerInfo.id);
    }
  }

  tick(ticks: number): void {
    if (this.player === null) {
      return;
    }
    this.pruneRecentNavalTargets(ticks);

    if (this.mg.inSpawnPhase()) {
      if (ticks % this.attackRate !== this.attackTick) {
        return;
      }

      if (this.nation.spawnCell === undefined) {
        this.mg.addExecution(
          new SpawnExecution(this.gameID, this.nation.playerInfo),
        );
        return;
      }

      const rl = this.randomSpawnLand();
      if (rl === null) {
        return;
      }

      this.mg.addExecution(
        new SpawnExecution(this.gameID, this.nation.playerInfo, rl),
      );
      return;
    }

    if (!this.player.isAlive()) {
      this.active = false;
      return;
    }

    const wakingUnderThreat = this.isThreatenedDuringStartup(ticks);
    if (ticks < ZOMBIE_START_DELAY_TICKS && !wakingUnderThreat) {
      return;
    }

    if (!wakingUnderThreat && ticks % this.attackRate !== this.attackTick) {
      return;
    }

    if (wakingUnderThreat) {
      this.launchLandAttacks({
        allowWithoutTrigger: true,
        attacksPerCycle: ZOMBIE_STARTUP_RETALIATION_ATTACKS_PER_CYCLE,
        maxFocusedTargets: 1,
        reserveRatio: 0,
        troopShare: ZOMBIE_STARTUP_RETALIATION_TROOP_SHARE,
        troopCap: ZOMBIE_MAX_STARTUP_RETALIATION_TROOPS,
        ticks,
      });
      return;
    }

    this.launchLandAttacks({ ticks });
    this.launchWaterAttacks(ticks);
    this.expandIntoOpenLand(ticks);
  }

  private launchLandAttacks(options?: {
    allowWithoutTrigger?: boolean;
    attacksPerCycle?: number;
    maxFocusedTargets?: number;
    reserveRatio?: number;
    troopShare?: number;
    troopCap?: number;
    ticks?: number;
  }): void {
    if (
      this.player === null ||
      (!options?.allowWithoutTrigger && !this.hasTriggerTroops())
    ) {
      return;
    }
    const player = this.player;
    const ticks = options?.ticks ?? this.mg.ticks();
    const maxFocusedTargets =
      options?.maxFocusedTargets ?? this.maxFocusedTargetsForTick(ticks);
    const reserveRatio = options?.reserveRatio ?? ZOMBIE_RESERVE_RATIO;
    const troopShare = options?.troopShare ?? ZOMBIE_LAND_ATTACK_TROOP_SHARE;
    const troopCap = options?.troopCap ?? this.landAttackTroopCapForTick(ticks);

    const activeOutgoingAttacks = player
      .outgoingAttacks()
      .filter((attack) => attack.isActive() && !attack.retreating());
    const activeLandAttacks = activeOutgoingAttacks
      .filter(
        (attack) =>
          attack.target().isPlayer(),
      );
    const remainingLandSlots = this.remainingLandPressureSlotsForTick(
      ticks,
      activeOutgoingAttacks.length,
    );
    const remainingPressureSlots = this.remainingTotalPressureSlotsForTick(
      ticks,
      activeOutgoingAttacks.length,
      player.unitCount(UnitType.TransportShip),
    );
    const remainingSlots =
      Math.min(
        remainingLandSlots,
        remainingPressureSlots,
      );
    if (remainingSlots <= 0) {
      return;
    }

    const activeAttacksByTarget = new Map<string, number>();
    for (const attack of activeLandAttacks) {
      const target = attack.target();
      if (!target.isPlayer()) {
        continue;
      }
      activeAttacksByTarget.set(
        target.id(),
        (activeAttacksByTarget.get(target.id()) ?? 0) + 1,
      );
    }

    const fronts = this.collectLandFronts(
      player,
      activeAttacksByTarget,
      maxFocusedTargets,
    );
    if (fronts.length === 0) {
      return;
    }

    const attacksToQueue = Math.min(
      options?.attacksPerCycle ?? this.landAttacksPerCycleForTick(ticks),
      remainingSlots,
      fronts.length,
    );
    if (attacksToQueue <= 0) {
      return;
    }

    let remainingTroopsBudget = Math.max(
      0,
      Math.floor(
        this.availableTroops(reserveRatio) *
          Math.min(1, troopShare * attacksToQueue),
      ),
    );
    if (remainingTroopsBudget < 1) {
      return;
    }

    let queuedAttacks = 0;
    for (const front of fronts) {
      const activeAgainstTarget =
        activeAttacksByTarget.get(front.enemy.id()) ?? 0;
      if (activeAgainstTarget >= ZOMBIE_MAX_ACTIVE_ATTACKS_PER_TARGET) {
        continue;
      }

      const troops = this.computeWaveTroopsForTarget(
        front.enemy,
        ticks,
        "land",
        remainingTroopsBudget,
        attacksToQueue - queuedAttacks,
        troopCap,
      );
      if (troops < 1) {
        continue;
      }

      this.mg.addExecution(
        new AttackExecution(troops, player, front.enemy.id(), front.sourceTile),
      );
      activeAttacksByTarget.set(front.enemy.id(), activeAgainstTarget + 1);
      remainingTroopsBudget = Math.max(0, remainingTroopsBudget - troops);
      queuedAttacks++;

      if (queuedAttacks >= attacksToQueue) {
        break;
      }
    }
  }

  private launchWaterAttacks(ticks: number): void {
    if (this.player === null || !this.hasTriggerTroops()) {
      return;
    }
    const player = this.player;
    const activeOutgoingAttacks = player
      .outgoingAttacks()
      .filter((attack) => attack.isActive() && !attack.retreating());
    const activeTransports = player.unitCount(UnitType.TransportShip);
    const remainingNavalSlots = this.remainingNavalPressureSlotsForTick(
      ticks,
      activeTransports,
    );
    const remainingPressureSlots = this.remainingTotalPressureSlotsForTick(
      ticks,
      activeOutgoingAttacks.length,
      activeTransports,
    );
    if (remainingPressureSlots <= 0 || remainingNavalSlots <= 0) {
      return;
    }

    const transportCap = Math.min(
      this.mg.config().boatMaxNumber(player),
      this.maxActiveTransportsForTick(ticks),
      activeTransports + remainingNavalSlots,
    );
    if (activeTransports >= transportCap) {
      return;
    }

    const targets = this.mg
      .players()
      .filter((other) => other !== player && !other.isFriendly(player))
      .filter((other) => this.shouldLaunchWaterAttackAt(player, other))
      .filter((other) => this.hasOceanShore(other))
      .sort((a, b) => this.compareWaterTargetPriority(player, a, b));

    if (targets.length === 0) {
      return;
    }

    const boatsToQueue = Math.min(
      this.waterAttacksPerCycleForTick(ticks),
      transportCap - activeTransports,
      remainingNavalSlots,
      remainingPressureSlots,
    );
    let remainingTroopsBudget = Math.max(
      0,
      Math.floor(
        this.availableTroops(ZOMBIE_RESERVE_RATIO) *
          Math.min(1, ZOMBIE_WATER_ATTACK_TROOP_SHARE * boatsToQueue),
      ),
    );
    if (remainingTroopsBudget < 1) {
      return;
    }

    const maxFocusedTargets = this.maxFocusedTargetsForTick(ticks);
    let boatsSent = 0;
    let attemptedTargets = 0;
    const primaryTarget = targets[0] ?? null;
    const shouldReinforcePrimaryTarget =
      primaryTarget !== null &&
      (this.hasRecentNavalPressure(primaryTarget) ||
        this.countSharedBorderFrontsAgainst(player, primaryTarget) <=
          ZOMBIE_NAVAL_REINFORCEMENT_FRONT_LIMIT);

    if (primaryTarget !== null && shouldReinforcePrimaryTarget) {
      while (boatsSent < boatsToQueue) {
        const targetShoreTile = this.findDeployableTargetShore(
          player,
          primaryTarget,
        );
        if (targetShoreTile === null) {
          break;
        }

        const troops = this.computeWaveTroopsForTarget(
          primaryTarget,
          ticks,
          "water",
          remainingTroopsBudget,
          boatsToQueue - boatsSent,
          this.waterAttackTroopCapForTick(ticks),
        );
        if (troops < 1) {
          break;
        }

        this.mg.addExecution(
          new TransportShipExecution(player, targetShoreTile, troops),
        );
        this.markRecentNavalPressure(primaryTarget, ticks);
        remainingTroopsBudget = Math.max(0, remainingTroopsBudget - troops);
        boatsSent++;
      }
    }

    for (const target of targets) {
      if (boatsSent >= boatsToQueue) {
        break;
      }
      if (shouldReinforcePrimaryTarget && target === primaryTarget) {
        continue;
      }
      if (attemptedTargets >= maxFocusedTargets) {
        break;
      }
      attemptedTargets++;
      if (remainingTroopsBudget < 1) {
        break;
      }

      const targetShoreTile = this.findDeployableTargetShore(player, target);
      if (targetShoreTile === null) {
        continue;
      }

      const troops = this.computeWaveTroopsForTarget(
        target,
        ticks,
        "water",
        remainingTroopsBudget,
        boatsToQueue - boatsSent,
        this.waterAttackTroopCapForTick(ticks),
      );
      if (troops < 1) {
        continue;
      }

      this.mg.addExecution(
        new TransportShipExecution(player, targetShoreTile, troops),
      );
      this.markRecentNavalPressure(target, ticks);
      remainingTroopsBudget = Math.max(0, remainingTroopsBudget - troops);
      boatsSent++;
    }
  }

  private expandIntoOpenLand(ticks: number): void {
    if (this.player === null) {
      return;
    }
    const player = this.player;
    let remainingLandPressureSlots = this.remainingLandPressureSlotsForTick(
      ticks,
      this.activeOutgoingAttackCount(player),
    );
    let remainingPressureSlots = this.remainingTotalPressureSlotsForTick(
      ticks,
      this.activeOutgoingAttackCount(player),
      player.unitCount(UnitType.TransportShip),
    );
    if (remainingPressureSlots <= 0 || remainingLandPressureSlots <= 0) {
      return;
    }

    const falloutFronts = this.collectOpenLandFronts(player, {
      includeFallout: true,
      includeOpenLand: false,
    }).slice(0, Math.min(remainingLandPressureSlots, remainingPressureSlots));
    const queuedFalloutAttacks = this.queueOpenLandAttacks(
      player,
      falloutFronts,
      0.18,
    );
    remainingLandPressureSlots = Math.max(
      0,
      remainingLandPressureSlots - queuedFalloutAttacks,
    );
    remainingPressureSlots = Math.max(0, remainingPressureSlots - queuedFalloutAttacks);

    if (this.hasEnemyBorder()) {
      return;
    }
    if (remainingPressureSlots <= 0 || remainingLandPressureSlots <= 0) {
      return;
    }

    const openLandFronts = this.collectOpenLandFronts(player, {
      includeFallout: false,
      includeOpenLand: true,
    }).slice(0, Math.min(remainingLandPressureSlots, remainingPressureSlots));
    this.queueOpenLandAttacks(player, openLandFronts, ZOMBIE_EXPAND_RATIO);
  }

  private queueOpenLandAttacks(
    player: Player,
    openLandFronts: TileRef[],
    troopShare: number,
  ): number {
    if (openLandFronts.length === 0) {
      return 0;
    }

    const troops = Math.max(
      1,
      Math.floor(
        this.availableTroops(troopShare) / openLandFronts.length,
      ),
    );
    if (troops < 1) {
      return 0;
    }

    for (const sourceTile of openLandFronts) {
      this.mg.addExecution(
        new AttackExecution(
          troops,
          player,
          this.mg.terraNullius().id(),
          sourceTile,
        ),
      );
    }
    return openLandFronts.length;
  }

  private availableTroops(reserveRatio: number): number {
    if (this.player === null) {
      return 0;
    }
    const reserve = this.mg.config().maxTroops(this.player) * reserveRatio;
    return Math.max(0, this.player.troops() - reserve);
  }

  private hasTriggerTroops(): boolean {
    if (this.player === null) {
      return false;
    }
    const maxTroops = this.mg.config().maxTroops(this.player);
    return this.player.troops() >= maxTroops * ZOMBIE_TRIGGER_RATIO;
  }

  private hasOceanShore(player: Player): boolean {
    for (const tile of player.borderTiles()) {
      if (this.mg.isOceanShore(tile)) {
        return true;
      }
    }
    return false;
  }

  private pruneRecentNavalTargets(ticks: number): void {
    for (const [targetID, expiresAt] of this.recentNavalTargets.entries()) {
      if (expiresAt <= ticks) {
        this.recentNavalTargets.delete(targetID);
      }
    }
  }

  private markRecentNavalPressure(target: Player, ticks: number): void {
    this.recentNavalTargets.set(
      target.id(),
      ticks + ZOMBIE_NAVAL_TARGET_MEMORY_TICKS,
    );
  }

  private hasRecentNavalPressure(target: Player): boolean {
    return this.recentNavalTargets.has(target.id());
  }

  private maxFocusedTargetsForTick(ticks: number): number {
    return this.rampZombieInt(
      ZOMBIE_MAX_FOCUSED_TARGETS,
      ZOMBIE_LATE_GAME_MAX_FOCUSED_TARGETS,
      ticks,
    );
  }

  private maxActiveLandAttacksForTick(ticks: number): number {
    return this.rampZombieInt(
      ZOMBIE_MAX_ACTIVE_LAND_ATTACKS,
      ZOMBIE_LATE_GAME_MAX_ACTIVE_LAND_ATTACKS,
      ticks,
    );
  }

  private landAttacksPerCycleForTick(ticks: number): number {
    return this.rampZombieInt(
      ZOMBIE_LAND_ATTACKS_PER_CYCLE,
      ZOMBIE_LATE_GAME_LAND_ATTACKS_PER_CYCLE,
      ticks,
    );
  }

  private maxActiveTransportsForTick(ticks: number): number {
    return this.rampZombieInt(
      ZOMBIE_MAX_ACTIVE_TRANSPORTS,
      ZOMBIE_LATE_GAME_MAX_ACTIVE_TRANSPORTS,
      ticks,
      ZOMBIE_ENDGAME_WAVE_RAMP_TICKS,
    );
  }

  private maxTotalPressureOperationsForTick(ticks: number): number {
    return this.rampZombieInt(
      ZOMBIE_MAX_TOTAL_PRESSURE_OPERATIONS,
      ZOMBIE_LATE_GAME_MAX_TOTAL_PRESSURE_OPERATIONS,
      ticks,
      ZOMBIE_ENDGAME_WAVE_RAMP_TICKS,
    );
  }

  private waterAttacksPerCycleForTick(ticks: number): number {
    return this.rampZombieInt(
      ZOMBIE_WATER_ATTACKS_PER_CYCLE,
      ZOMBIE_LATE_GAME_WATER_ATTACKS_PER_CYCLE,
      ticks,
      ZOMBIE_ENDGAME_WAVE_RAMP_TICKS,
    );
  }

  private landAttackTroopCapForTick(ticks: number): number {
    return this.rampZombieInt(
      ZOMBIE_MAX_LAND_ATTACK_TROOPS,
      ZOMBIE_LATE_GAME_MAX_LAND_ATTACK_TROOPS,
      ticks,
      ZOMBIE_ENDGAME_WAVE_RAMP_TICKS,
    );
  }

  private waterAttackTroopCapForTick(ticks: number): number {
    return this.rampZombieInt(
      ZOMBIE_MAX_WATER_ATTACK_TROOPS,
      ZOMBIE_LATE_GAME_MAX_WATER_ATTACK_TROOPS,
      ticks,
      ZOMBIE_ENDGAME_WAVE_RAMP_TICKS,
    );
  }

  private rampZombieInt(
    base: number,
    max: number,
    ticks: number,
    rampTicks = ZOMBIE_FOCUSED_TARGET_RAMP_TICKS,
  ): number {
    if (ticks <= ZOMBIE_START_DELAY_TICKS || max <= base) {
      return base;
    }

    const elapsedTicks = ticks - ZOMBIE_START_DELAY_TICKS;
    const progress = Math.min(1, elapsedTicks / rampTicks);
    return base + Math.floor(progress * (max - base) + 1e-9);
  }

  private rampZombieFloat(
    base: number,
    max: number,
    ticks: number,
    rampTicks = ZOMBIE_ENDGAME_WAVE_RAMP_TICKS,
  ): number {
    if (ticks <= ZOMBIE_START_DELAY_TICKS || max <= base) {
      return base;
    }

    const elapsedTicks = ticks - ZOMBIE_START_DELAY_TICKS;
    const progress = Math.min(1, elapsedTicks / rampTicks);
    return base + (max - base) * progress;
  }

  private activeOutgoingAttackCount(player: Player): number {
    return player
      .outgoingAttacks()
      .filter((attack) => attack.isActive() && !attack.retreating()).length;
  }

  private remainingLandPressureSlotsForTick(
    ticks: number,
    activeOutgoingAttacks: number,
  ): number {
    return Math.max(
      0,
      this.maxActiveLandAttacksForTick(ticks) - activeOutgoingAttacks,
    );
  }

  private remainingNavalPressureSlotsForTick(
    ticks: number,
    activeTransports: number,
  ): number {
    return Math.max(
      0,
      this.maxActiveTransportsForTick(ticks) - activeTransports,
    );
  }

  private remainingTotalPressureSlotsForTick(
    ticks: number,
    activeOutgoingAttacks: number,
    activeTransports: number,
  ): number {
    return Math.max(
      0,
      this.maxTotalPressureOperationsForTick(ticks) -
        activeOutgoingAttacks -
        activeTransports,
    );
  }

  private computeWaveTimeScale(ticks: number): number {
    const elapsedTicks = Math.max(0, ticks - ZOMBIE_START_DELAY_TICKS);
    return Math.min(
      ZOMBIE_WAVE_TIME_SCALE_MAX,
      1 + elapsedTicks / ZOMBIE_WAVE_TIME_SCALE_TICKS,
    );
  }

  private computeWaveTroopsForTarget(
    target: Player,
    ticks: number,
    waveKind: "land" | "water",
    remainingTroopsBudget: number,
    remainingWaves: number,
    troopCap: number,
  ): number {
    if (remainingTroopsBudget < 1 || remainingWaves < 1) {
      return 0;
    }

    const targetMaxTroops = this.mg.config().maxTroops(target);
    const waveTimeScale = this.computeWaveTimeScale(ticks);
    const minTroops =
      waveKind === "land"
        ? ZOMBIE_MIN_LAND_ATTACK_TROOPS
        : ZOMBIE_MIN_WATER_ATTACK_TROOPS;
    const targetRatio =
      waveKind === "land"
        ? ZOMBIE_LAND_ATTACK_TARGET_MAX_TROOPS_RATIO
        : ZOMBIE_WATER_ATTACK_TARGET_MAX_TROOPS_RATIO;
    const targetRatioScale = this.rampZombieFloat(1, 1.5, ticks);

    let desiredTroops = Math.max(
      minTroops,
      Math.floor(targetMaxTroops * targetRatio * targetRatioScale * waveTimeScale),
    );
    if (this.hasRecentNavalPressure(target)) {
      desiredTroops = Math.floor(desiredTroops * ZOMBIE_BEACHHEAD_WAVE_BONUS);
    }

    const reserveForRemainingWaves = Math.max(
      0,
      minTroops * (remainingWaves - 1),
    );
    const maxAllocatable =
      remainingTroopsBudget > reserveForRemainingWaves
        ? remainingTroopsBudget - reserveForRemainingWaves
        : Math.max(1, Math.floor(remainingTroopsBudget / remainingWaves));

    return Math.max(
      1,
      Math.min(
        troopCap,
        desiredTroops,
        maxAllocatable,
        remainingTroopsBudget,
      ),
    );
  }

  private shouldLaunchWaterAttackAt(player: Player, target: Player): boolean {
    if (!player.sharesBorderWith(target)) {
      return true;
    }
    if (this.hasRecentNavalPressure(target)) {
      return true;
    }
    return (
      this.countSharedBorderFrontsAgainst(player, target) <=
      ZOMBIE_NAVAL_REINFORCEMENT_FRONT_LIMIT
    );
  }

  private compareWaterTargetPriority(
    player: Player,
    a: Player,
    b: Player,
  ): number {
    const recentPressureDiff =
      Number(this.hasRecentNavalPressure(b)) -
      Number(this.hasRecentNavalPressure(a));
    if (recentPressureDiff !== 0) {
      return recentPressureDiff;
    }

    const frontDiff =
      this.countSharedBorderFrontsAgainst(player, a) -
      this.countSharedBorderFrontsAgainst(player, b);
    if (frontDiff !== 0) {
      return frontDiff;
    }

    return a.troops() - b.troops();
  }

  private countSharedBorderFrontsAgainst(
    player: Player,
    target: Player,
  ): number {
    let fronts = 0;
    for (const tile of player.borderTiles()) {
      const touchesTarget = this.mg
        .neighbors(tile)
        .some(
          (neighbor) =>
            this.mg.isLand(neighbor) &&
            this.mg.hasOwner(neighbor) &&
            this.mg.ownerID(neighbor) === target.smallID(),
        );
      if (!touchesTarget) {
        continue;
      }
      fronts++;
      if (fronts > ZOMBIE_NAVAL_REINFORCEMENT_FRONT_LIMIT) {
        break;
      }
    }
    return fronts;
  }

  private collectLandFronts(
    player: Player,
    activeAttacksByTarget: Map<string, number>,
    maxFocusedTargets: number,
  ): Array<{ enemy: Player; sourceTile: TileRef }> {
    const frontsByEnemy = new Map<
      string,
      { enemy: Player; sourceTiles: Set<TileRef> }
    >();

    const borderTiles = this.random
      .shuffleArray(Array.from(player.borderTiles()))
      .slice(0, ZOMBIE_FRONT_SCAN_LIMIT);
    for (const tile of borderTiles) {
      const adjacentEnemies = new Set<Player>();
      for (const neighbor of this.mg.neighbors(tile)) {
        if (!this.mg.isLand(neighbor) || !this.mg.hasOwner(neighbor)) {
          continue;
        }
        const owner = this.mg.owner(neighbor);
        if (!owner.isPlayer() || owner === player || owner.isFriendly(player)) {
          continue;
        }
        adjacentEnemies.add(owner);
      }

      for (const enemy of adjacentEnemies) {
        const existing = frontsByEnemy.get(enemy.id());
        if (existing) {
          existing.sourceTiles.add(tile);
        } else {
          frontsByEnemy.set(enemy.id(), {
            enemy,
            sourceTiles: new Set([tile]),
          });
        }
      }
    }

    const sortedEnemies = Array.from(frontsByEnemy.values()).sort((a, b) => {
      const recentPressureDiff =
        Number(this.hasRecentNavalPressure(b.enemy)) -
        Number(this.hasRecentNavalPressure(a.enemy));
      if (recentPressureDiff !== 0) {
        return recentPressureDiff;
      }

      const activePressureDiff =
        (activeAttacksByTarget.get(b.enemy.id()) ?? 0) -
        (activeAttacksByTarget.get(a.enemy.id()) ?? 0);
      if (activePressureDiff !== 0) {
        return activePressureDiff;
      }

      const frontierSizeDiff = b.sourceTiles.size - a.sourceTiles.size;
      if (frontierSizeDiff !== 0) {
        return frontierSizeDiff;
      }

      return a.enemy.troops() - b.enemy.troops();
    });
    const focusedEnemies = sortedEnemies.slice(0, maxFocusedTargets);

    const fronts: Array<{ enemy: Player; sourceTile: TileRef }> = [];
    for (const { enemy, sourceTiles } of focusedEnemies) {
      const shuffledSourceTiles = this.random
        .shuffleArray(Array.from(sourceTiles))
        .slice(0, ZOMBIE_FRONTS_PER_TARGET);
      for (const sourceTile of shuffledSourceTiles) {
        fronts.push({ enemy, sourceTile });
      }
    }
    return fronts;
  }

  private hasEnemyBorder(): boolean {
    if (this.player === null) {
      return false;
    }

    return Array.from(this.player.borderTiles()).some((tile) =>
      this.mg
        .neighbors(tile)
        .some(
          (neighbor) =>
            this.mg.isLand(neighbor) &&
            this.mg.hasOwner(neighbor) &&
            this.mg.owner(neighbor).isPlayer() &&
            this.mg.owner(neighbor) !== this.player,
        ),
    );
  }

  private isThreatenedDuringStartup(ticks: number): boolean {
    return ticks < ZOMBIE_START_DELAY_TICKS && this.hasEnemyBorder();
  }

  private collectOpenLandFronts(
    player: Player,
    options?: {
      includeFallout?: boolean;
      includeOpenLand?: boolean;
    },
  ): TileRef[] {
    const fronts: TileRef[] = [];
    const includeFallout = options?.includeFallout ?? true;
    const includeOpenLand = options?.includeOpenLand ?? true;
    const borderTiles = this.random
      .shuffleArray(Array.from(player.borderTiles()))
      .slice(0, ZOMBIE_FRONT_SCAN_LIMIT);
    for (const tile of borderTiles) {
      const hasOpenNeighbor = this.mg
        .neighbors(tile)
        .some(
          (neighbor) =>
            this.mg.isLand(neighbor) &&
            !this.mg.hasOwner(neighbor) &&
            ((includeOpenLand && !this.mg.hasFallout(neighbor)) ||
              (includeFallout && this.mg.hasFallout(neighbor))),
        );
      if (!hasOpenNeighbor) {
        continue;
      }

      fronts.push(tile);
      if (fronts.length >= ZOMBIE_OPEN_LAND_ATTACKS_PER_CYCLE) {
        break;
      }
    }
    return fronts;
  }

  private findDeployableTargetShore(
    player: Player,
    target: Player,
  ): TileRef | null {
    const sharedFrontTargetTiles = this.collectSharedFrontTargetTiles(
      player,
      target,
    );
    const scanLimit =
      sharedFrontTargetTiles.length > 0
        ? ZOMBIE_TARGET_SHORE_SCAN_LIMIT * 3
        : ZOMBIE_TARGET_SHORE_SCAN_LIMIT;
    const candidateTiles: TileRef[] = [];
    let scannedShoreTiles = 0;
    for (const tile of this.random.shuffleArray(
      Array.from(target.borderTiles()),
    )) {
      if (!this.mg.isOceanShore(tile)) {
        continue;
      }
      scannedShoreTiles++;
      if (bestShoreDeploymentSource(this.mg, player, tile) !== null) {
        candidateTiles.push(tile);
      }
      if (scannedShoreTiles >= scanLimit) {
        break;
      }
    }
    if (candidateTiles.length === 0) {
      return null;
    }
    if (sharedFrontTargetTiles.length === 0) {
      return candidateTiles[0];
    }

    candidateTiles.sort(
      (a, b) =>
        this.closestDistanceToTiles(a, sharedFrontTargetTiles) -
        this.closestDistanceToTiles(b, sharedFrontTargetTiles),
    );
    return candidateTiles[0];
  }

  private collectSharedFrontTargetTiles(
    player: Player,
    target: Player,
  ): TileRef[] {
    const frontTiles: TileRef[] = [];
    for (const tile of target.borderTiles()) {
      const touchesZombie = this.mg
        .neighbors(tile)
        .some(
          (neighbor) =>
            this.mg.isLand(neighbor) &&
            this.mg.hasOwner(neighbor) &&
            this.mg.owner(neighbor) === player,
        );
      if (touchesZombie) {
        frontTiles.push(tile);
      }
    }
    return frontTiles;
  }

  private closestDistanceToTiles(tile: TileRef, refs: TileRef[]): number {
    let minDistance = Infinity;
    for (const ref of refs) {
      minDistance = Math.min(minDistance, this.mg.manhattanDist(tile, ref));
    }
    return minDistance;
  }

  private randomSpawnLand(): TileRef | null {
    if (this.nation.spawnCell === undefined) {
      throw new Error("Zombie nation spawn cell missing");
    }

    const delta = 25;
    let tries = 0;
    while (tries < 50) {
      tries++;
      const cell = this.nation.spawnCell;
      const x = this.random.nextInt(cell.x - delta, cell.x + delta);
      const y = this.random.nextInt(cell.y - delta, cell.y + delta);
      if (!this.mg.isValidCoord(x, y)) {
        continue;
      }
      const tile = this.mg.ref(x, y);
      if (this.mg.isLand(tile) && !this.mg.hasOwner(tile)) {
        if (
          this.mg.terrainType(tile) === TerrainType.Mountain &&
          this.random.chance(2)
        ) {
          continue;
        }
        return tile;
      }
    }
    return null;
  }

  isActive(): boolean {
    return this.active;
  }

  activeDuringSpawnPhase(): boolean {
    return true;
  }
}
