import path from "path";
import { describe, expect, it, vi } from "vitest";
import { DefaultConfig } from "../src/core/configuration/DefaultConfig";
import { AttackExecution } from "../src/core/execution/AttackExecution";
import { PlayerExecution } from "../src/core/execution/PlayerExecution";
import { ShellExecution } from "../src/core/execution/ShellExecution";
import { WarshipExecution } from "../src/core/execution/WarshipExecution";
import { ZombieNationExecution } from "../src/core/execution/ZombieNationExecution";
import { ZombieSurvivalExecution } from "../src/core/execution/ZombieSurvivalExecution";
import {
  CureState,
  GameMapSize,
  GameMode,
  GameType,
  Nation,
  PlayerInfo,
  PlayerSpecialRole,
  PlayerType,
  UnitType,
} from "../src/core/game/Game";
import { createNationsForGame } from "../src/core/game/NationCreation";
import {
  ZOMBIE_ENDGAME_WAVE_RAMP_TICKS,
  ZOMBIE_FOCUSED_TARGET_RAMP_TICKS,
  ZOMBIE_LATE_GAME_LAND_ATTACKS_PER_CYCLE,
  ZOMBIE_LATE_GAME_MAX_ACTIVE_LAND_ATTACKS,
  ZOMBIE_LATE_GAME_MAX_ACTIVE_TRANSPORTS,
  ZOMBIE_LATE_GAME_MAX_FOCUSED_TARGETS,
  ZOMBIE_LATE_GAME_MAX_LAND_ATTACK_TROOPS,
  ZOMBIE_LATE_GAME_MAX_TOTAL_PRESSURE_OPERATIONS,
  ZOMBIE_MAX_ATTACK_STEPS_PER_TICK,
  ZOMBIE_MAX_FOCUSED_TARGETS,
  ZOMBIE_MAX_LAND_ATTACK_TROOPS,
  ZOMBIE_MAX_TOTAL_PRESSURE_OPERATIONS,
  ZOMBIE_MAX_TERRA_NULLIUS_ATTACK_STEPS_PER_TICK,
  ZOMBIE_MAX_WATER_ATTACK_TROOPS,
  ZOMBIE_START_DELAY_TICKS,
  ZOMBIE_UNCURED_ATTACKER_LOSS_MULTIPLIER,
  ZOMBIE_UNCURED_ATTACK_INTERVAL_TICKS,
  ZOMBIE_UNCURED_DEFENSE_FLOOR_PER_TILE,
} from "../src/core/game/ZombieUtils";
import { PseudoRandom } from "../src/core/PseudoRandom";
import { setup } from "./util/Setup";
import { TestConfig, UseRealAttackLogic } from "./util/TestConfig";

function getLandTiles(game: any): number[] {
  const tiles: number[] = [];
  game.map().forEachTile((tile: number) => {
    if (game.map().isLand(tile)) {
      tiles.push(tile);
    }
  });
  return tiles;
}

function getWaterTiles(game: any): number[] {
  const tiles: number[] = [];
  game.map().forEachTile((tile: number) => {
    if (game.map().isWater(tile)) {
      tiles.push(tile);
    }
  });
  return tiles;
}

function getAdjacentLandChain(game: any, length: number): number[] {
  const visited = new Set<number>();
  const path: number[] = [];

  const dfs = (tile: number): boolean => {
    visited.add(tile);
    path.push(tile);
    if (path.length === length) {
      return true;
    }

    for (const neighbor of game.neighbors(tile)) {
      if (!game.isLand(neighbor) || visited.has(neighbor)) {
        continue;
      }
      if (dfs(neighbor)) {
        return true;
      }
    }

    path.pop();
    visited.delete(tile);
    return false;
  };

  for (const tile of getLandTiles(game)) {
    visited.clear();
    path.length = 0;
    if (dfs(tile)) {
      return [...path];
    }
  }

  throw new Error(`Unable to find adjacent land chain of length ${length}`);
}

function getLandTileWithLandNeighbors(
  game: any,
  minNeighbors: number,
): { center: number; neighbors: number[] } {
  for (const tile of getLandTiles(game)) {
    const landNeighbors = game
      .neighbors(tile)
      .filter((neighbor: number) => game.isLand(neighbor));
    if (landNeighbors.length >= minNeighbors) {
      return {
        center: tile,
        neighbors: landNeighbors.slice(0, minNeighbors),
      };
    }
  }

  throw new Error(`Unable to find land tile with ${minNeighbors} neighbors`);
}

function advanceToZombieVulnerable(game: any): void {
  while (game.ticks() < ZOMBIE_START_DELAY_TICKS) {
    game.executeNextTick();
  }
}

class UseRealZombieCombatConfig extends UseRealAttackLogic {
  attackTilesPerTick(
    attackTroops: number,
    attacker: any,
    defender: any,
    numAdjacentTilesWithEnemy: number,
  ): number {
    return DefaultConfig.prototype.attackTilesPerTick.call(
      this,
      attackTroops,
      attacker,
      defender,
      numAdjacentTilesWithEnemy,
    );
  }
}

class ExtremeAttackBudgetConfig extends TestConfig {
  attackLogic(
    _gm: any,
    _attackTroops: number,
    _attacker: any,
    _defender: any,
    _tileToConquer: number,
  ) {
    return {
      attackerTroopLoss: 0,
      defenderTroopLoss: 0,
      tilesPerTickUsed: 0,
    };
  }

  attackTilesPerTick(
    _attackTroops: number,
    _attacker: any,
    _defender: any,
    _numAdjacentTilesWithEnemy: number,
  ): number {
    return Number.MAX_SAFE_INTEGER;
  }
}

describe("Zombie Survival", () => {
  it("replaces exactly one nation with the zombie nation in zombie rulesets", () => {
    const random = new PseudoRandom(7);
    const nations = createNationsForGame(
      {
        gameID: "zombie-test",
        players: [],
        config: {
          gameMap: "Asia",
          gameMapSize: GameMapSize.Normal,
          gameMode: GameMode.FFA,
          gameType: GameType.Public,
          difficulty: "Medium",
          nations: "default",
          donateGold: false,
          donateTroops: false,
          bots: 0,
          infiniteGold: false,
          infiniteTroops: false,
          instantBuild: false,
          randomSpawn: false,
          specialRuleset: "zombie_survival",
        },
      } as any,
      [
        { name: "A", coordinates: [1, 1] },
        { name: "B", coordinates: [2, 2] },
        { name: "C", coordinates: [3, 3] },
      ] as any,
      0,
      random,
    );

    expect(nations).toHaveLength(3);
    expect(
      nations.filter(
        (nation) => nation.playerInfo.specialRole === PlayerSpecialRole.Zombie,
      ),
    ).toHaveLength(1);
  });

  it("applies zombie-only troop cap, regen, and boat cap overrides", async () => {
    const game = await setup("big_plains", {
      specialRuleset: "zombie_survival",
      infiniteGold: true,
      instantBuild: true,
    });

    const regularNation = game.addPlayer(
      new PlayerInfo("Nation", PlayerType.Nation, null, "nation"),
    );
    const zombieNation = game.addPlayer(
      new PlayerInfo(
        "Zombie Nation",
        PlayerType.Nation,
        null,
        "zombie",
        false,
        null,
        PlayerSpecialRole.Zombie,
      ),
    );

    const landTiles = getLandTiles(game);
    for (let i = 0; i < 8; i++) {
      regularNation.conquer(landTiles[i]);
      zombieNation.conquer(landTiles[i + 16]);
    }
    regularNation.setTroops(20_000);
    zombieNation.setTroops(20_000);

    const regularMax = game.config().maxTroops(regularNation);
    const zombieMax = game.config().maxTroops(zombieNation);
    const regularRegen = game.config().troopIncreaseRate(regularNation);
    const zombieRegen = game.config().troopIncreaseRate(zombieNation);

    expect(zombieMax).toBe(regularMax * 8);
    expect(zombieRegen).toBeGreaterThan(regularRegen * 11.9);
    expect(game.config().boatMaxNumber(zombieNation)).toBe(30);
    expect(game.config().boatMaxNumber(regularNation)).toBe(3);

    zombieNation.setTroops(Math.floor(zombieMax * 0.6));
    expect(game.config().troopIncreaseRate(zombieNation)).toBeGreaterThan(0);
  });

  it("prices labs at 5M in FFA and 25M in teams", async () => {
    const ffaGame = await setup("big_plains", {
      specialRuleset: "zombie_survival",
      gameMode: GameMode.FFA,
    });
    const teamGame = await setup("big_plains", {
      specialRuleset: "zombie_survival",
      gameMode: GameMode.Team,
      playerTeams: 2,
    });

    expect(
      ffaGame
        .config()
        .unitInfo(UnitType.ResearchLab)
        .cost(
          ffaGame,
          ffaGame.addPlayer(
            new PlayerInfo("FFA Human", PlayerType.Human, null, "ffa-human"),
          ),
        ),
    ).toBe(5_000_000n);
    expect(
      teamGame
        .config()
        .unitInfo(UnitType.ResearchLab)
        .cost(
          teamGame,
          teamGame.addPlayer(
            new PlayerInfo("Team Human", PlayerType.Human, null, "team-human"),
          ),
        ),
    ).toBe(25_000_000n);
  });

  it("makes uncured attacks into zombie land brutally inefficient", async () => {
    const game = await setup(
      "big_plains",
      {
        specialRuleset: "zombie_survival",
        infiniteGold: true,
        instantBuild: true,
      },
      [],
      path.join(__dirname, "util"),
      UseRealZombieCombatConfig,
    );

    const human = game.addPlayer(
      new PlayerInfo("Human", PlayerType.Human, null, "human"),
    );
    const zombie = game.addPlayer(
      new PlayerInfo(
        "Zombie Nation",
        PlayerType.Nation,
        null,
        "zombie",
        false,
        null,
        PlayerSpecialRole.Zombie,
      ),
    );

    const chain = getAdjacentLandChain(game, 4);
    human.conquer(chain[0]);
    zombie.conquer(chain[1]);
    zombie.conquer(chain[2]);
    zombie.conquer(chain[3]);
    human.setTroops(100_000);
    zombie.setTroops(60_000);
    for (let tick = 0; tick <= game.config().numSpawnPhaseTurns(); tick++) {
      game.executeNextTick();
    }
    advanceToZombieVulnerable(game);

    const attack = new AttackExecution(60_000, human, zombie.id());
    const currentTick = game.ticks();
    const nextZombieProgressTick =
      currentTick % ZOMBIE_UNCURED_ATTACK_INTERVAL_TICKS === 0
        ? currentTick
        : currentTick +
          (ZOMBIE_UNCURED_ATTACK_INTERVAL_TICKS -
            (currentTick % ZOMBIE_UNCURED_ATTACK_INTERVAL_TICKS));
    attack.init(game, currentTick);

    const startingZombieTiles = zombie.numTilesOwned();
    for (let tick = currentTick; tick < nextZombieProgressTick; tick++) {
      attack.tick(tick);
    }
    expect(zombie.numTilesOwned()).toBe(startingZombieTiles);

    attack.tick(nextZombieProgressTick);
    expect(zombie.numTilesOwned()).toBeLessThan(startingZombieTiles);

    const uncuredOutcome = game
      .config()
      .attackLogic(game, 20_000, human, zombie, chain[2]);

    human.setCureState(CureState.Cured);
    const curedOutcome = game
      .config()
      .attackLogic(game, 20_000, human, zombie, chain[2]);

    expect(uncuredOutcome.attackerTroopLoss).toBeGreaterThan(
      curedOutcome.attackerTroopLoss * 100,
    );
    expect(uncuredOutcome.attackerTroopLoss).toBeGreaterThanOrEqual(
      ZOMBIE_UNCURED_DEFENSE_FLOOR_PER_TILE *
        ZOMBIE_UNCURED_ATTACKER_LOSS_MULTIPLIER,
    );
    expect(uncuredOutcome.defenderTroopLoss).toBeLessThan(
      curedOutcome.defenderTroopLoss / 100,
    );
  });

  it("prevents uncured players from auto-annexing zombies under 100 tiles", async () => {
    const game = await setup("big_plains", {
      specialRuleset: "zombie_survival",
      infiniteGold: true,
      instantBuild: true,
    });

    const human = game.addPlayer(
      new PlayerInfo("Human", PlayerType.Human, null, "human"),
    );
    const zombie = game.addPlayer(
      new PlayerInfo(
        "Zombie Nation",
        PlayerType.Nation,
        null,
        "zombie",
        false,
        null,
        PlayerSpecialRole.Zombie,
      ),
    );

    const chain = getAdjacentLandChain(game, 3);
    human.conquer(chain[0]);
    zombie.conquer(chain[1]);
    zombie.conquer(chain[2]);
    human.setTroops(100_000);
    zombie.setTroops(10);

    while (game.inSpawnPhase()) {
      game.executeNextTick();
    }
    advanceToZombieVulnerable(game);

    const attack = new AttackExecution(100_000, human, zombie.id());
    const nextZombieProgressTick =
      game.ticks() % ZOMBIE_UNCURED_ATTACK_INTERVAL_TICKS === 0
        ? game.ticks()
        : game.ticks() +
          (ZOMBIE_UNCURED_ATTACK_INTERVAL_TICKS -
            (game.ticks() % ZOMBIE_UNCURED_ATTACK_INTERVAL_TICKS));
    attack.init(game, game.ticks());
    attack.tick(nextZombieProgressTick);

    expect(zombie.isAlive()).toBe(true);
    expect(zombie.numTilesOwned()).toBe(1);

    human.setCureState(CureState.Cured);
    const curedAttack = new AttackExecution(100_000, human, zombie.id());
    curedAttack.init(game, game.ticks());
    curedAttack.tick(game.ticks());

    expect(zombie.isAlive()).toBe(false);
    expect(game.getWinner()).toBeNull();
  });

  it("lets cured players instantly collapse large zombie empires without inheriting them", async () => {
    const game = await setup("big_plains", {
      specialRuleset: "zombie_survival",
      infiniteGold: true,
      instantBuild: true,
    });

    const human = game.addPlayer(
      new PlayerInfo("Human", PlayerType.Human, null, "human"),
    );
    const rival = game.addPlayer(
      new PlayerInfo("Rival", PlayerType.Nation, null, "rival"),
    );
    const zombie = game.addPlayer(
      new PlayerInfo(
        "Zombie Nation",
        PlayerType.Nation,
        null,
        "zombie",
        false,
        null,
        PlayerSpecialRole.Zombie,
      ),
    );

    const breach = getAdjacentLandChain(game, 2);
    human.conquer(breach[0]);
    zombie.conquer(breach[1]);

    const allLandTiles = getLandTiles(game);
    const extraZombieTiles = allLandTiles
      .filter((tile) => !breach.includes(tile))
      .slice(0, 120);
    for (const tile of extraZombieTiles) {
      zombie.conquer(tile);
    }

    const rivalTiles = allLandTiles
      .filter(
        (tile) =>
          !breach.includes(tile) && !extraZombieTiles.includes(tile),
      )
      .slice(0, 3);
    for (const tile of rivalTiles) {
      rival.conquer(tile);
    }

    human.setTroops(200_000);
    zombie.setTroops(50_000);

    while (game.inSpawnPhase()) {
      game.executeNextTick();
    }
    advanceToZombieVulnerable(game);

    human.setCureState(CureState.Cured);

    const attack = new AttackExecution(100_000, human, zombie.id());
    attack.init(game, game.ticks());
    attack.tick(game.ticks());

    const survivalExecution = new ZombieSurvivalExecution();
    survivalExecution.init(game);
    survivalExecution.tick(game.ticks());

    expect(zombie.isAlive()).toBe(false);
    expect(zombie.numTilesOwned()).toBe(0);
    expect(human.numTilesOwned()).toBe(2);
    expect(game.owner(extraZombieTiles[extraZombieTiles.length - 1]).isPlayer()).toBe(
      false,
    );
    expect(game.getWinner()).toBe(rival);
  });

  it("waits one minute before zombie aggression begins", async () => {
    const game = await setup("big_plains", {
      specialRuleset: "zombie_survival",
      infiniteGold: true,
      instantBuild: true,
    });

    while (game.inSpawnPhase()) {
      game.executeNextTick();
    }

    const human = game.addPlayer(
      new PlayerInfo("Human", PlayerType.Human, null, "human"),
    );
    const zombieInfo = new PlayerInfo(
      "Zombie Nation",
      PlayerType.Nation,
      null,
      "zombie",
      false,
      null,
      PlayerSpecialRole.Zombie,
    );
    const zombie = game.addPlayer(zombieInfo);

    const chain = getAdjacentLandChain(game, 3);
    zombie.conquer(chain[0]);
    human.conquer(chain[2]);
    zombie.setTroops(game.config().maxTroops(zombie));

    const execution = new ZombieNationExecution(
      "zombie-delay-test",
      new Nation(undefined, zombieInfo),
    );
    execution.init(game);

    for (let tick = game.ticks(); tick < ZOMBIE_START_DELAY_TICKS; tick++) {
      execution.tick(tick);
    }
    expect((game as any).executions()).toHaveLength(0);

    let aggressionQueued = false;
    for (
      let tick = ZOMBIE_START_DELAY_TICKS;
      tick < ZOMBIE_START_DELAY_TICKS + 50;
      tick++
    ) {
      execution.tick(tick);
      if ((game as any).executions().length > 0) {
        aggressionQueued = true;
        break;
      }
    }

    expect(aggressionQueued).toBe(true);
    const queuedLandAttack = (game as any)
      .executions()
      .find((queued: any) => queued instanceof AttackExecution);
    expect(queuedLandAttack).toBeDefined();
    queuedLandAttack.init(game, ZOMBIE_START_DELAY_TICKS);
    expect(zombie.outgoingAttacks()).toHaveLength(1);
    expect(zombie.outgoingAttacks()[0].sourceTile()).not.toBeNull();
  });

  it("makes the zombie nation unattackable during its startup delay", async () => {
    const game = await setup("big_plains", {
      specialRuleset: "zombie_survival",
      infiniteGold: true,
      instantBuild: true,
    });

    while (game.inSpawnPhase()) {
      game.executeNextTick();
    }

    const human = game.addPlayer(
      new PlayerInfo("Human", PlayerType.Human, null, "human"),
    );
    const bot = game.addPlayer(
      new PlayerInfo("Bot", PlayerType.Bot, null, "bot"),
    );
    const zombie = game.addPlayer(
      new PlayerInfo(
        "Zombie Nation",
        PlayerType.Nation,
        null,
        "zombie",
        false,
        null,
        PlayerSpecialRole.Zombie,
      ),
    );

    expect(game.ticks()).toBeLessThan(ZOMBIE_START_DELAY_TICKS);
    expect(human.canAttackPlayer(zombie)).toBe(false);
    expect(bot.canAttackPlayer(zombie)).toBe(false);

    while (game.ticks() < ZOMBIE_START_DELAY_TICKS) {
      game.executeNextTick();
    }

    expect(human.canAttackPlayer(zombie)).toBe(true);
    expect(bot.canAttackPlayer(zombie)).toBe(true);
  });

  it("wakes the zombie early and counterattacks if surrounded during startup", async () => {
    const game = await setup("big_plains", {
      specialRuleset: "zombie_survival",
      infiniteGold: true,
      instantBuild: true,
    });

    while (game.inSpawnPhase()) {
      game.executeNextTick();
    }

    const human = game.addPlayer(
      new PlayerInfo("Human", PlayerType.Human, null, "human"),
    );
    const zombieInfo = new PlayerInfo(
      "Zombie Nation",
      PlayerType.Nation,
      null,
      "zombie",
      false,
      null,
      PlayerSpecialRole.Zombie,
    );
    const zombie = game.addPlayer(zombieInfo);

    const chain = getAdjacentLandChain(game, 2);
    zombie.conquer(chain[0]);
    human.conquer(chain[1]);
    zombie.setTroops(game.config().maxTroops(zombie));

    expect(game.ticks()).toBeLessThan(ZOMBIE_START_DELAY_TICKS);

    const execution = new ZombieNationExecution(
      "zombie-startup-retaliation-test",
      new Nation(undefined, zombieInfo),
    );
    execution.init(game);
    execution.tick(game.ticks());

    const queuedLandAttack = (game as any)
      .executions()
      .find((queued: any) => queued instanceof AttackExecution);
    expect(queuedLandAttack).toBeDefined();
    queuedLandAttack.init(game, game.ticks());
    expect(zombie.outgoingAttacks()).toHaveLength(1);
    expect(zombie.outgoingAttacks()[0].target().id()).toBe(human.id());
  });

  it("does not let surrounded zombie land get annexed by cluster cleanup", async () => {
    const game = await setup("big_plains", {
      specialRuleset: "zombie_survival",
      infiniteGold: true,
      instantBuild: true,
    });

    while (game.inSpawnPhase()) {
      game.executeNextTick();
    }

    const human = game.addPlayer(
      new PlayerInfo("Human", PlayerType.Human, null, "human"),
    );
    const zombie = game.addPlayer(
      new PlayerInfo(
        "Zombie Nation",
        PlayerType.Nation,
        null,
        "zombie",
        false,
        null,
        PlayerSpecialRole.Zombie,
      ),
    );

    const { center, neighbors } = getLandTileWithLandNeighbors(game, 4);
    zombie.conquer(center);
    neighbors.forEach((tile) => human.conquer(tile));

    advanceToZombieVulnerable(game);

    const execution = new PlayerExecution(zombie);
    execution.init(game, game.ticks());
    execution.tick(game.ticks());

    expect(zombie.isAlive()).toBe(true);
    expect(zombie.numTilesOwned()).toBe(1);
    expect(game.owner(center)).toBe(zombie);
  });

  it("keeps zombie aggression focused on a small number of enemy nations", async () => {
    const game = await setup("big_plains", {
      specialRuleset: "zombie_survival",
      infiniteGold: true,
      instantBuild: true,
    });

    while (game.inSpawnPhase()) {
      game.executeNextTick();
    }

    const zombieInfo = new PlayerInfo(
      "Zombie Nation",
      PlayerType.Nation,
      null,
      "zombie",
      false,
      null,
      PlayerSpecialRole.Zombie,
    );
    const zombie = game.addPlayer(zombieInfo);
    const enemies = Array.from({ length: 4 }, (_, index) =>
      game.addPlayer(
        new PlayerInfo(
          `Enemy ${index + 1}`,
          PlayerType.Human,
          null,
          `enemy-${index + 1}`,
        ),
      ),
    );

    const { center, neighbors } = getLandTileWithLandNeighbors(game, 4);
    zombie.conquer(center);
    zombie.setTroops(game.config().maxTroops(zombie));
    enemies.forEach((enemy, index) => {
      enemy.conquer(neighbors[index]);
      enemy.setTroops(25_000);
    });

    const execution = new ZombieNationExecution(
      "zombie-focus-test",
      new Nation(undefined, zombieInfo),
    );
    execution.init(game);

    let queuedLandAttacks: AttackExecution[] = [];
    for (
      let tick = ZOMBIE_START_DELAY_TICKS;
      tick < ZOMBIE_START_DELAY_TICKS + 80;
      tick++
    ) {
      execution.tick(tick);
      queuedLandAttacks = (game as any)
        .executions()
        .filter((queued: any) => queued instanceof AttackExecution);
      if (queuedLandAttacks.length > 0) {
        break;
      }
    }

    expect(queuedLandAttacks.length).toBeGreaterThan(0);
    const uniqueTargetCount = new Set(
      queuedLandAttacks.map((attack) => attack.targetID()),
    ).size;
    expect(uniqueTargetCount).toBeLessThanOrEqual(ZOMBIE_MAX_FOCUSED_TARGETS);
  });

  it("caps single zombie land waves even with a massive reserve", async () => {
    const game = await setup("big_plains", {
      specialRuleset: "zombie_survival",
      infiniteGold: true,
      instantBuild: true,
    });

    while (game.inSpawnPhase()) {
      game.executeNextTick();
    }

    const human = game.addPlayer(
      new PlayerInfo("Human", PlayerType.Human, null, "human"),
    );
    const zombieInfo = new PlayerInfo(
      "Zombie Nation",
      PlayerType.Nation,
      null,
      "zombie",
      false,
      null,
      PlayerSpecialRole.Zombie,
    );
    const zombie = game.addPlayer(zombieInfo);

    const chain = getAdjacentLandChain(game, 2);
    zombie.conquer(chain[0]);
    human.conquer(chain[1]);
    zombie.setTroops(game.config().maxTroops(zombie));

    const execution = new ZombieNationExecution(
      "zombie-wave-cap-test",
      new Nation(undefined, zombieInfo),
    );
    execution.init(game);

    let queuedLandAttack: AttackExecution | undefined;
    for (
      let tick = ZOMBIE_START_DELAY_TICKS;
      tick < ZOMBIE_START_DELAY_TICKS + 50;
      tick++
    ) {
      execution.tick(tick);
      queuedLandAttack = (game as any)
        .executions()
        .find((queued: any) => queued instanceof AttackExecution);
      if (queuedLandAttack) {
        break;
      }
    }

    expect(queuedLandAttack).toBeDefined();
    queuedLandAttack!.init(game, ZOMBIE_START_DELAY_TICKS);
    expect(zombie.outgoingAttacks().length).toBeGreaterThan(0);
    expect(zombie.outgoingAttacks()[0].troops()).toBeLessThanOrEqual(
      ZOMBIE_MAX_LAND_ATTACK_TROOPS,
    );
  });

  it("scales zombie wave size with target strength and elapsed time", () => {
    const zombieInfo = new PlayerInfo(
      "Zombie Nation",
      PlayerType.Nation,
      null,
      "zombie",
      false,
      null,
      PlayerSpecialRole.Zombie,
    );
    const execution = new ZombieNationExecution(
      "zombie-wave-scaling-test",
      new Nation(undefined, zombieInfo),
    ) as any;

    execution.mg = {
      ticks: () => ZOMBIE_START_DELAY_TICKS,
      config: () => ({
        maxTroops: (player: { id: () => string }) =>
          player.id() === "large" ? 1_200_000 : 240_000,
      }),
    };

    const smallTarget = { id: () => "small" };
    const largeTarget = { id: () => "large" };

    const earlySmallWave = execution.computeWaveTroopsForTarget(
      smallTarget,
      ZOMBIE_START_DELAY_TICKS,
      "land",
      1_000_000,
      1,
      ZOMBIE_MAX_LAND_ATTACK_TROOPS,
    );
    const lateSmallWave = execution.computeWaveTroopsForTarget(
      smallTarget,
      ZOMBIE_START_DELAY_TICKS + 6_000,
      "land",
      1_000_000,
      1,
      ZOMBIE_MAX_LAND_ATTACK_TROOPS,
    );
    const earlyLargeWave = execution.computeWaveTroopsForTarget(
      largeTarget,
      ZOMBIE_START_DELAY_TICKS,
      "land",
      1_000_000,
      1,
      ZOMBIE_MAX_LAND_ATTACK_TROOPS,
    );
    const lateLargeBoatWave = execution.computeWaveTroopsForTarget(
      largeTarget,
      ZOMBIE_START_DELAY_TICKS + 6_000,
      "water",
      1_000_000,
      1,
      ZOMBIE_MAX_WATER_ATTACK_TROOPS,
    );

    expect(earlyLargeWave).toBeGreaterThan(earlySmallWave);
    expect(earlyLargeWave).toBeGreaterThan(100_000);
    expect(lateSmallWave).toBeGreaterThan(earlySmallWave);
    expect(lateLargeBoatWave).toBeLessThanOrEqual(
      ZOMBIE_MAX_WATER_ATTACK_TROOPS,
    );
    expect(lateLargeBoatWave).toBeGreaterThan(earlySmallWave);
  });

  it("focuses repeated boat reinforcements onto the current invasion target", () => {
    const zombieInfo = new PlayerInfo(
      "Zombie Nation",
      PlayerType.Nation,
      null,
      "zombie",
      false,
      null,
      PlayerSpecialRole.Zombie,
    );
    const execution = new ZombieNationExecution(
      "zombie-water-focus-test",
      new Nation(undefined, zombieInfo),
    ) as any;

    const addExecution = vi.fn();
    const zombiePlayer = {
      id: () => "zombie-player",
      unitCount: vi.fn(() => 0),
      outgoingAttacks: vi.fn(() => []),
      isFriendly: vi.fn(() => false),
      sharesBorderWith: vi.fn(
        (other: { id: () => string }) => other.id() === "target-a",
      ),
      borderTiles: vi.fn(() => new Set([1])),
    };
    const targetA = {
      id: () => "target-a",
      smallID: () => 2,
      isFriendly: vi.fn(() => false),
      borderTiles: vi.fn(() => new Set([10, 11])),
      troops: vi.fn(() => 50_000),
    };
    const targetB = {
      id: () => "target-b",
      smallID: () => 3,
      isFriendly: vi.fn(() => false),
      borderTiles: vi.fn(() => new Set([20, 21])),
      troops: vi.fn(() => 55_000),
    };

    execution.player = zombiePlayer;
    execution.mg = {
      config: () => ({
        boatMaxNumber: () => 30,
        maxTroops: (player: { id: () => string }) =>
          player.id() === "target-a" ? 800_000 : 500_000,
      }),
      players: () => [zombiePlayer, targetA, targetB],
      addExecution,
      isOceanShore: vi.fn(() => true),
    };
    execution.hasTriggerTroops = vi.fn(() => true);
    execution.availableTroops = vi.fn(() => 500_000);
    execution.findDeployableTargetShore = vi.fn(
      (_player: unknown, target: { id: () => string }) =>
        target.id() === "target-a" ? 10 : 20,
    );
    execution.countSharedBorderFrontsAgainst = vi.fn(
      (_player: unknown, target: { id: () => string }) =>
        target.id() === "target-a" ? 1 : 6,
    );
    execution.markRecentNavalPressure(targetA, 100);

    execution.launchWaterAttacks(100);

    expect(addExecution).toHaveBeenCalledTimes(2);
    expect(execution.findDeployableTargetShore).toHaveBeenNthCalledWith(
      1,
      zombiePlayer,
      targetA,
    );
    expect(execution.findDeployableTargetShore).toHaveBeenNthCalledWith(
      2,
      zombiePlayer,
      targetA,
    );
  });

  it("ramps focused targets from 2 to 24 by the fifteen minute mark", () => {
    const zombieInfo = new PlayerInfo(
      "Zombie Nation",
      PlayerType.Nation,
      null,
      "zombie",
      false,
      null,
      PlayerSpecialRole.Zombie,
    );
    const execution = new ZombieNationExecution(
      "zombie-late-focus-test",
      new Nation(undefined, zombieInfo),
    ) as any;

    expect(execution.maxFocusedTargetsForTick(ZOMBIE_START_DELAY_TICKS)).toBe(
      ZOMBIE_MAX_FOCUSED_TARGETS,
    );
    expect(
      execution.maxFocusedTargetsForTick(
        ZOMBIE_START_DELAY_TICKS + ZOMBIE_FOCUSED_TARGET_RAMP_TICKS / 3,
      ),
    ).toBe(9);
    expect(
      execution.maxFocusedTargetsForTick(
        ZOMBIE_START_DELAY_TICKS + (ZOMBIE_FOCUSED_TARGET_RAMP_TICKS * 2) / 3,
      ),
    ).toBe(16);
    expect(
      execution.maxFocusedTargetsForTick(
        ZOMBIE_START_DELAY_TICKS + ZOMBIE_FOCUSED_TARGET_RAMP_TICKS,
      ),
    ).toBe(ZOMBIE_LATE_GAME_MAX_FOCUSED_TARGETS);
  });

  it("ramps late-game land pressure alongside the wider target spread", () => {
    const zombieInfo = new PlayerInfo(
      "Zombie Nation",
      PlayerType.Nation,
      null,
      "zombie",
      false,
      null,
      PlayerSpecialRole.Zombie,
    );
    const execution = new ZombieNationExecution(
      "zombie-land-pressure-ramp-test",
      new Nation(undefined, zombieInfo),
    ) as any;

    expect(
      execution.maxActiveLandAttacksForTick(ZOMBIE_START_DELAY_TICKS),
    ).toBe(8);
    expect(
      execution.landAttacksPerCycleForTick(ZOMBIE_START_DELAY_TICKS),
    ).toBe(4);
    expect(
      execution.maxActiveLandAttacksForTick(
        ZOMBIE_START_DELAY_TICKS + ZOMBIE_FOCUSED_TARGET_RAMP_TICKS,
      ),
    ).toBe(ZOMBIE_LATE_GAME_MAX_ACTIVE_LAND_ATTACKS);
    expect(
      execution.landAttacksPerCycleForTick(
        ZOMBIE_START_DELAY_TICKS + ZOMBIE_FOCUSED_TARGET_RAMP_TICKS,
      ),
    ).toBe(ZOMBIE_LATE_GAME_LAND_ATTACKS_PER_CYCLE);
    expect(
      execution.maxTotalPressureOperationsForTick(ZOMBIE_START_DELAY_TICKS),
    ).toBe(ZOMBIE_MAX_TOTAL_PRESSURE_OPERATIONS);
    expect(
      execution.maxTotalPressureOperationsForTick(
        ZOMBIE_START_DELAY_TICKS + ZOMBIE_ENDGAME_WAVE_RAMP_TICKS,
      ),
    ).toBe(ZOMBIE_LATE_GAME_MAX_TOTAL_PRESSURE_OPERATIONS);
  });

  it("keeps land pressure separate from saturated boat pressure", () => {
    const zombieInfo = new PlayerInfo(
      "Zombie Nation",
      PlayerType.Nation,
      null,
      "zombie",
      false,
      null,
      PlayerSpecialRole.Zombie,
    );
    const execution = new ZombieNationExecution(
      "zombie-pressure-saturation-test",
      new Nation(undefined, zombieInfo),
    ) as any;

    const player = {
      outgoingAttacks: () => [],
      unitCount: () => ZOMBIE_LATE_GAME_MAX_ACTIVE_TRANSPORTS,
    };
    const enemy = {
      id: () => "enemy",
    };

    execution.player = player;
    execution.mg = {
      ticks: () => ZOMBIE_START_DELAY_TICKS + ZOMBIE_ENDGAME_WAVE_RAMP_TICKS,
      config: () => ({
        maxTroops: () => 800_000,
      }),
      addExecution: vi.fn(),
    };
    execution.hasTriggerTroops = vi.fn(() => true);
    execution.availableTroops = vi.fn(() => 500_000);
    execution.collectLandFronts = vi.fn(() => [{ enemy, sourceTile: 1 }]);

    execution.launchLandAttacks({
      ticks: ZOMBIE_START_DELAY_TICKS + ZOMBIE_ENDGAME_WAVE_RAMP_TICKS,
    });

    expect(execution.collectLandFronts).toHaveBeenCalled();
    expect(execution.mg.addExecution).toHaveBeenCalledTimes(1);
  });

  it("keeps boat pressure separate from saturated land pressure", () => {
    const zombieInfo = new PlayerInfo(
      "Zombie Nation",
      PlayerType.Nation,
      null,
      "zombie",
      false,
      null,
      PlayerSpecialRole.Zombie,
    );
    const execution = new ZombieNationExecution(
      "zombie-naval-pressure-separation-test",
      new Nation(undefined, zombieInfo),
    ) as any;

    const activeAttack = {
      isActive: () => true,
      retreating: () => false,
      target: () => ({ isPlayer: () => true, id: () => "enemy" }),
    };
    const zombiePlayer = {
      id: () => "zombie-player",
      outgoingAttacks: () =>
        Array.from(
          { length: ZOMBIE_LATE_GAME_MAX_ACTIVE_LAND_ATTACKS },
          () => activeAttack,
        ),
      unitCount: vi.fn(() => 0),
      isFriendly: vi.fn(() => false),
      sharesBorderWith: vi.fn(() => false),
      borderTiles: vi.fn(() => new Set([1])),
    };
    const target = {
      id: () => "target-a",
      smallID: () => 2,
      isFriendly: vi.fn(() => false),
      borderTiles: vi.fn(() => new Set([10, 11])),
      troops: vi.fn(() => 50_000),
    };

    execution.player = zombiePlayer;
    execution.mg = {
      config: () => ({
        boatMaxNumber: () => 30,
        maxTroops: () => 800_000,
      }),
      players: () => [zombiePlayer, target],
      addExecution: vi.fn(),
      isOceanShore: vi.fn(() => true),
    };
    execution.hasTriggerTroops = vi.fn(() => true);
    execution.availableTroops = vi.fn(() => 500_000);
    execution.findDeployableTargetShore = vi.fn(() => 10);
    execution.countSharedBorderFrontsAgainst = vi.fn(() => 0);

    execution.launchWaterAttacks(
      ZOMBIE_START_DELAY_TICKS + ZOMBIE_ENDGAME_WAVE_RAMP_TICKS,
    );

    expect(execution.mg.addExecution).toHaveBeenCalled();
  });

  it("keeps reinforcing a fresh naval beachhead before it becomes a full front", () => {
    const zombieInfo = new PlayerInfo(
      "Zombie Nation",
      PlayerType.Nation,
      null,
      "zombie",
      false,
      null,
      PlayerSpecialRole.Zombie,
    );
    const execution = new ZombieNationExecution(
      "zombie-naval-pressure-test",
      new Nation(undefined, zombieInfo),
    ) as any;
    const target = {
      id: () => "enemy",
      smallID: () => 7,
    };
    const player = {
      sharesBorderWith: vi.fn(() => true),
      borderTiles: vi.fn(() => new Set([1, 2])),
    };

    execution.mg = {
      neighbors: vi.fn((tile: number) => [tile + 100]),
      isLand: vi.fn(() => true),
      hasOwner: vi.fn(() => true),
      ownerID: vi.fn((tile: number) => (tile === 101 ? 7 : 0)),
    };

    expect(execution.shouldLaunchWaterAttackAt(player, target)).toBe(true);

    player.borderTiles.mockReturnValue(new Set([1, 2, 3, 4, 5, 6]));
    execution.mg.ownerID.mockReturnValue(7);
    expect(execution.shouldLaunchWaterAttackAt(player, target)).toBe(false);

    execution.markRecentNavalPressure(target, 100);
    expect(execution.shouldLaunchWaterAttackAt(player, target)).toBe(true);
  });

  it("uses local expansion attacks when spreading into open land", async () => {
    const game = await setup("big_plains", {
      specialRuleset: "zombie_survival",
      infiniteGold: true,
      instantBuild: true,
    });

    while (game.inSpawnPhase()) {
      game.executeNextTick();
    }

    const zombieInfo = new PlayerInfo(
      "Zombie Nation",
      PlayerType.Nation,
      null,
      "zombie",
      false,
      null,
      PlayerSpecialRole.Zombie,
    );
    const zombie = game.addPlayer(zombieInfo);

    const [zombieTile] = getAdjacentLandChain(game, 2);
    zombie.conquer(zombieTile);
    zombie.setTroops(game.config().maxTroops(zombie));

    const execution = new ZombieNationExecution(
      "zombie-expand-test",
      new Nation(undefined, zombieInfo),
    );
    execution.init(game);

    let queuedExpansion: AttackExecution | undefined;
    for (
      let tick = ZOMBIE_START_DELAY_TICKS;
      tick < ZOMBIE_START_DELAY_TICKS + 80;
      tick++
    ) {
      execution.tick(tick);
      queuedExpansion = (game as any)
        .executions()
        .find(
          (queued: any) =>
            queued instanceof AttackExecution &&
            queued.targetID() === game.terraNullius().id(),
        );
      if (queuedExpansion) {
        break;
      }
    }

    expect(queuedExpansion).toBeDefined();
    queuedExpansion!.init(game, ZOMBIE_START_DELAY_TICKS);
    expect(zombie.outgoingAttacks()).toHaveLength(1);
    expect(zombie.outgoingAttacks()[0].sourceTile()).not.toBeNull();
  });

  it("reclaims adjacent fallout even while zombies are actively bordering a living nation", async () => {
    const game = await setup("big_plains", {
      specialRuleset: "zombie_survival",
      infiniteGold: true,
      instantBuild: true,
    });

    while (game.inSpawnPhase()) {
      game.executeNextTick();
    }

    const zombieInfo = new PlayerInfo(
      "Zombie Nation",
      PlayerType.Nation,
      null,
      "zombie",
      false,
      null,
      PlayerSpecialRole.Zombie,
    );
    const zombie = game.addPlayer(zombieInfo);
    const human = game.addPlayer(
      new PlayerInfo("Human", PlayerType.Human, null, "human"),
    );

    const { center, neighbors } = getLandTileWithLandNeighbors(game, 2);
    const enemyTile = neighbors[0];
    const falloutTile = neighbors[1];

    zombie.conquer(center);
    zombie.setTroops(game.config().maxTroops(zombie));
    human.conquer(enemyTile);
    game.setFallout(falloutTile, true);

    const execution = new ZombieNationExecution(
      "zombie-fallout-reclaim-test",
      new Nation(undefined, zombieInfo),
    );
    execution.init(game);

    let queuedExpansion: AttackExecution | undefined;
    for (
      let tick = ZOMBIE_START_DELAY_TICKS;
      tick < ZOMBIE_START_DELAY_TICKS + 80;
      tick++
    ) {
      execution.tick(tick);
      queuedExpansion = (game as any)
        .executions()
        .find(
          (queued: any) =>
            queued instanceof AttackExecution &&
            queued.targetID() === game.terraNullius().id(),
        );
      if (queuedExpansion) {
        break;
      }
    }

    expect(queuedExpansion).toBeDefined();
  });

  it("lets zombies overrun fallout far more easily than normal attackers", async () => {
    const game = await setup(
      "big_plains",
      {
        specialRuleset: "zombie_survival",
        infiniteGold: true,
        instantBuild: true,
      },
      [],
      path.join(__dirname, "util"),
      UseRealAttackLogic,
    );

    while (game.inSpawnPhase()) {
      game.executeNextTick();
    }

    const zombie = game.addPlayer(
      new PlayerInfo(
        "Zombie Nation",
        PlayerType.Nation,
        null,
        "zombie",
        false,
        null,
        PlayerSpecialRole.Zombie,
      ),
    );
    const human = game.addPlayer(
      new PlayerInfo("Human", PlayerType.Human, null, "human"),
    );
    const [falloutTile] = getLandTiles(game);
    game.setFallout(falloutTile, true);

    const zombieResult = game.config().attackLogic(
      game,
      100_000,
      zombie,
      game.terraNullius(),
      falloutTile,
    );
    const humanResult = game.config().attackLogic(
      game,
      100_000,
      human,
      game.terraNullius(),
      falloutTile,
    );

    expect(zombieResult.attackerTroopLoss).toBeLessThan(
      humanResult.attackerTroopLoss,
    );
    expect(zombieResult.tilesPerTickUsed).toBeLessThan(
      humanResult.tilesPerTickUsed,
    );
  });

  it("caps giant attack workloads so one attack cannot lock a tick", async () => {
    const game = await setup(
      "big_plains",
      {
        specialRuleset: "zombie_survival",
        infiniteGold: true,
        instantBuild: true,
      },
      [],
      path.join(__dirname, "util"),
      ExtremeAttackBudgetConfig,
    );

    const attacker = game.addPlayer(
      new PlayerInfo("Attacker", PlayerType.Human, null, "attacker"),
    );
    const defender = game.addPlayer(
      new PlayerInfo(
        "Defender Nation",
        PlayerType.Nation,
        null,
        "stress-defender",
      ),
    );

    const chain = getAdjacentLandChain(game, 2);
    const attackerTile = chain[0];
    const allLandTiles = getLandTiles(game);

    attacker.conquer(attackerTile);
    for (const tile of allLandTiles) {
      if (tile !== attackerTile) {
        defender.conquer(tile);
      }
    }

    attacker.setTroops(1_000_000);
    defender.setTroops(1_000_000);

    const startingDefenderTiles = defender.numTilesOwned();
    expect(startingDefenderTiles).toBeGreaterThan(5_000);

    while (game.inSpawnPhase()) {
      game.executeNextTick();
    }

    const attack = new AttackExecution(1_000_000, attacker, defender.id());
    attack.init(game, game.ticks());
    attack.tick(game.ticks());

    expect(defender.numTilesOwned()).toBeLessThan(startingDefenderTiles);
    expect(defender.numTilesOwned()).toBeGreaterThan(1_000);
    expect(attack.isActive()).toBe(true);
  });

  it("applies a lower per-tick workload cap to zombie attacks", async () => {
    const game = await setup(
      "big_plains",
      {
        specialRuleset: "zombie_survival",
        infiniteGold: true,
        instantBuild: true,
      },
      [],
      path.join(__dirname, "util"),
      ExtremeAttackBudgetConfig,
    );

    const zombie = game.addPlayer(
      new PlayerInfo(
        "Zombie Nation",
        PlayerType.Nation,
        null,
        "zombie",
        false,
        null,
        PlayerSpecialRole.Zombie,
      ),
    );
    const defender = game.addPlayer(
      new PlayerInfo("Defender Nation", PlayerType.Nation, null, "defender"),
    );

    const chain = getAdjacentLandChain(game, 2);
    const zombieTile = chain[0];
    const allLandTiles = getLandTiles(game);

    zombie.conquer(zombieTile);
    for (const tile of allLandTiles) {
      if (tile !== zombieTile) {
        defender.conquer(tile);
      }
    }

    zombie.setTroops(1_000_000);
    defender.setTroops(1_000_000);

    while (game.inSpawnPhase()) {
      game.executeNextTick();
    }

    const startingDefenderTiles = defender.numTilesOwned();
    const attack = new AttackExecution(1_000_000, zombie, defender.id());
    attack.init(game, game.ticks());
    attack.tick(game.ticks());

    expect(startingDefenderTiles - defender.numTilesOwned()).toBeLessThanOrEqual(
      ZOMBIE_MAX_ATTACK_STEPS_PER_TICK,
    );

    const falloutGame = await setup(
      "big_plains",
      {
        specialRuleset: "zombie_survival",
        infiniteGold: true,
        instantBuild: true,
      },
      [],
      path.join(__dirname, "util"),
      ExtremeAttackBudgetConfig,
    );

    const falloutZombie = falloutGame.addPlayer(
      new PlayerInfo(
        "Zombie Nation",
        PlayerType.Nation,
        null,
        "zombie-fallout",
        false,
        null,
        PlayerSpecialRole.Zombie,
      ),
    );
    const falloutDefender = falloutGame.addPlayer(
      new PlayerInfo(
        "Fallout Defender Nation",
        PlayerType.Nation,
        null,
        "fallout-defender",
      ),
    );

    const falloutChain = getAdjacentLandChain(falloutGame, 2);
    const falloutZombieTile = falloutChain[0];
    const falloutTile = falloutChain[1];

    falloutZombie.conquer(falloutZombieTile);
    falloutDefender.conquer(falloutTile);
    falloutZombie.setTroops(1_000_000);
    falloutDefender.setTroops(1_000_000);

    while (falloutGame.inSpawnPhase()) {
      falloutGame.executeNextTick();
    }

    falloutDefender.relinquish(falloutTile);
    falloutGame.setFallout(falloutTile, true);
    const falloutAttack = new AttackExecution(
      1_000_000,
      falloutZombie,
      falloutGame.terraNullius().id(),
      falloutZombieTile,
    );
    falloutAttack.init(falloutGame, falloutGame.ticks());
    falloutAttack.tick(falloutGame.ticks());

    expect(falloutZombie.numTilesOwned()).toBeLessThanOrEqual(
      1 + ZOMBIE_MAX_TERRA_NULLIUS_ATTACK_STEPS_PER_TICK,
    );
  });

  it("keeps zombie wave caps growing into true endgame", () => {
    const zombieInfo = new PlayerInfo(
      "Zombie Nation",
      PlayerType.Nation,
      null,
      "zombie",
      false,
      null,
      PlayerSpecialRole.Zombie,
    );
    const execution = new ZombieNationExecution(
      "zombie-endgame-wave-growth-test",
      new Nation(undefined, zombieInfo),
    ) as any;

    execution.mg = {
      config: () => ({
        maxTroops: (player: { id: () => string }) =>
          player.id() === "large" ? 2_000_000 : 250_000,
      }),
    };

    const largeTarget = { id: () => "large" };
    const earlyCap = execution.landAttackTroopCapForTick(ZOMBIE_START_DELAY_TICKS);
    const endgameCap = execution.landAttackTroopCapForTick(
      ZOMBIE_START_DELAY_TICKS + ZOMBIE_ENDGAME_WAVE_RAMP_TICKS,
    );
    const earlyWave = execution.computeWaveTroopsForTarget(
      largeTarget,
      ZOMBIE_START_DELAY_TICKS,
      "land",
      5_000_000,
      1,
      earlyCap,
    );
    const endgameWave = execution.computeWaveTroopsForTarget(
      largeTarget,
      ZOMBIE_START_DELAY_TICKS + ZOMBIE_ENDGAME_WAVE_RAMP_TICKS,
      "land",
      5_000_000,
      1,
      endgameCap,
    );

    expect(earlyCap).toBe(ZOMBIE_MAX_LAND_ATTACK_TROOPS);
    expect(endgameCap).toBe(ZOMBIE_LATE_GAME_MAX_LAND_ATTACK_TROOPS);
    expect(endgameWave).toBeGreaterThan(earlyWave);
    expect(endgameWave).toBeGreaterThan(ZOMBIE_MAX_LAND_ATTACK_TROOPS);
  });

  it("researches the cure and ends the match when zombies are gone", async () => {
    const game = await setup("big_plains", {
      specialRuleset: "zombie_survival",
      infiniteGold: true,
      instantBuild: true,
    });

    const human = game.addPlayer(
      new PlayerInfo("Human", PlayerType.Human, null, "human"),
    );
    const zombie = game.addPlayer(
      new PlayerInfo(
        "Zombie Nation",
        PlayerType.Nation,
        null,
        "zombie",
        false,
        null,
        PlayerSpecialRole.Zombie,
      ),
    );

    const landTiles = getLandTiles(game);
    for (let i = 0; i < 5; i++) {
      human.conquer(landTiles[i]);
    }
    zombie.conquer(landTiles[8]);
    zombie.conquer(landTiles[9]);

    human.buildUnit(UnitType.ResearchLab, landTiles[0], {});
    human.setTroops(100_000);

    const execution = new ZombieSurvivalExecution();
    execution.init(game);
    execution.tick(0);

    expect(human.cureState()).toBe(CureState.Researching);
    expect(human.cureProgressRemainingTicks()).toBe(5999);

    for (let tick = 1; tick <= 5999; tick++) {
      execution.tick(tick);
    }

    expect(human.cureState()).toBe(CureState.Cured);
    expect(human.cureProgressRemainingTicks()).toBeNull();

    human.conquer(landTiles[8]);
    human.conquer(landTiles[9]);
    execution.tick(6000);

    expect(game.getWinner()).toBe(human);
    expect(execution.isActive()).toBe(false);
    expect(zombie.isAlive()).toBe(false);
  });

  it("damages non-cured warships that kill zombie transports but spares cured warships", async () => {
    const game = await setup("half_land_half_ocean", {
      specialRuleset: "zombie_survival",
      infiniteGold: true,
      instantBuild: true,
    });

    const human = game.addPlayer(
      new PlayerInfo("Human", PlayerType.Human, null, "human"),
    );
    const zombie = game.addPlayer(
      new PlayerInfo(
        "Zombie Nation",
        PlayerType.Nation,
        null,
        "zombie",
        false,
        null,
        PlayerSpecialRole.Zombie,
      ),
    );

    const landTiles = getLandTiles(game);
    human.conquer(landTiles[0]);
    zombie.conquer(landTiles[1]);

    const waterTiles = getWaterTiles(game);
    const warship = human.buildUnit(UnitType.Warship, waterTiles[0], {
      patrolTile: waterTiles[0],
    });
    const transport = zombie.buildUnit(UnitType.TransportShip, waterTiles[0], {
      troops: 100,
      targetTile: landTiles[0],
    });

    const deathBurst = new ShellExecution(
      waterTiles[0],
      human,
      warship,
      transport,
    );
    deathBurst.init(game, 0);
    for (let tick = 0; tick < 10 && deathBurst.isActive(); tick++) {
      deathBurst.tick(tick);
    }

    expect(warship.health()).toBe(900);

    human.setCureState(CureState.Cured);
    const curedWarship = human.buildUnit(UnitType.Warship, waterTiles[1], {
      patrolTile: waterTiles[1],
    });
    const secondTransport = zombie.buildUnit(
      UnitType.TransportShip,
      waterTiles[1],
      {
        troops: 100,
        targetTile: landTiles[0],
      },
    );

    const immuneBurst = new ShellExecution(
      waterTiles[1],
      human,
      curedWarship,
      secondTransport,
    );
    immuneBurst.init(game, 0);
    for (let tick = 0; tick < 10 && immuneBurst.isActive(); tick++) {
      immuneBurst.tick(tick);
    }

    expect(curedWarship.health()).toBe(1000);
  });

  it("cleans up cleanly when zombie death-burst sinks the killing warship", async () => {
    const game = await setup("half_land_half_ocean", {
      specialRuleset: "zombie_survival",
      infiniteGold: true,
      instantBuild: true,
    });

    while (game.inSpawnPhase()) {
      game.executeNextTick();
    }

    const human = game.addPlayer(
      new PlayerInfo("Human", PlayerType.Human, null, "human"),
    );
    const zombie = game.addPlayer(
      new PlayerInfo(
        "Zombie Nation",
        PlayerType.Nation,
        null,
        "zombie",
        false,
        null,
        PlayerSpecialRole.Zombie,
      ),
    );

    const landTiles = getLandTiles(game);
    human.conquer(landTiles[0]);
    zombie.conquer(landTiles[1]);

    const waterTiles = getWaterTiles(game);
    const warship = human.buildUnit(UnitType.Warship, waterTiles[0], {
      patrolTile: waterTiles[0],
    });
    warship.modifyHealth(-900);

    const transport = zombie.buildUnit(UnitType.TransportShip, waterTiles[0], {
      troops: 100,
      targetTile: landTiles[0],
    });

    game.addExecution(new WarshipExecution(warship));
    advanceToZombieVulnerable(game);
    for (let tick = 0; tick < 10; tick++) {
      game.executeNextTick();
    }

    expect(warship.isActive()).toBe(false);
    expect(transport.isActive()).toBe(false);
    expect((game as any).executions()).toHaveLength(0);
    expect(
      game.units(UnitType.Shell).filter((unit) => unit.isActive()),
    ).toHaveLength(0);
  });
});
