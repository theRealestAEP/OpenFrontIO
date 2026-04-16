import { ConstructionExecution } from "../../../src/core/execution/ConstructionExecution";
import { OilExecution } from "../../../src/core/execution/OilExecution";
import { TransportShipExecution } from "../../../src/core/execution/TransportShipExecution";
import {
  Game,
  Player,
  PlayerInfo,
  PlayerType,
  UnitType,
} from "../../../src/core/game/Game";
import { TileRef } from "../../../src/core/game/GameMap";
import { oilRigPlacementMinDist } from "../../../src/core/game/OilRigUtils";
import { GameConfig } from "../../../src/core/Schemas";
import { setup } from "../../util/Setup";
import { executeTicks } from "../../util/utils";

async function createOilGame(
  mapName = "half_land_half_ocean",
  gameConfig: Partial<GameConfig> = {},
  players = [new PlayerInfo("player", PlayerType.Human, null, "player_id")],
): Promise<{ game: Game; players: Player[] }> {
  const game = await setup(mapName, gameConfig, players);

  while (game.inSpawnPhase()) {
    game.executeNextTick();
  }

  game.addExecution(new OilExecution());

  const resolvedPlayers = players.map((playerInfo) => {
    const player = game.player(playerInfo.id);
    player.addGold(10_000_000n);
    return player;
  });

  return { game, players: resolvedPlayers };
}

function firstLandTile(game: Game): TileRef {
  for (let y = 0; y < game.height(); y++) {
    for (let x = 0; x < game.width(); x++) {
      const tile = game.ref(x, y);
      if (game.isLand(tile)) {
        return tile;
      }
    }
  }
  throw new Error("Expected at least one land tile on the test map");
}

function firstActiveOceanOilTile(game: Game): TileRef {
  for (const field of game.oilFields()) {
    if (field.remainingReserve <= 0) {
      continue;
    }
    const tile = field.tiles.find((candidate) => game.isOcean(candidate));
    if (tile !== undefined) {
      return tile;
    }
  }
  throw new Error("Expected at least one ocean oil tile on the test map");
}

function nearbyOceanOilTileWithinSpacing(
  game: Game,
  origin: TileRef,
): TileRef | null {
  const maxDistSquared = oilRigPlacementMinDist(game) ** 2;
  const field = game.oilFieldAt(origin);
  if (field === null) {
    return null;
  }

  for (const tile of field.tiles) {
    if (
      tile !== origin &&
      game.isOcean(tile) &&
      game.euclideanDistSquared(origin, tile) < maxDistSquared
    ) {
      return tile;
    }
  }

  return null;
}

function overlappingFieldsAt(game: Game, tile: TileRef) {
  return game.oilFields().filter((field) => field.tiles.includes(tile));
}

function reachablePortTiles(game: Game, oceanTile: TileRef): TileRef[] {
  const minDistSquared = game.config().structureMinDist() ** 2;
  const candidates: TileRef[] = [];

  for (let y = 0; y < game.height(); y++) {
    for (let x = 0; x < game.width(); x++) {
      const tile = game.ref(x, y);
      if (!game.isOceanShore(tile)) {
        continue;
      }
      if (game.getWaterComponent(tile) !== game.getWaterComponent(oceanTile)) {
        continue;
      }
      if (game.euclideanDistSquared(tile, oceanTile) < minDistSquared) {
        continue;
      }
      candidates.push(tile);
    }
  }

  return candidates.sort(
    (a, b) =>
      game.manhattanDist(a, oceanTile) - game.manhattanDist(b, oceanTile),
  );
}

function buildReachablePort(
  game: Game,
  player: Player,
  oceanTile: TileRef,
  index = 0,
  blockedTiles: TileRef[] = [],
) {
  const minDistSquared = game.config().structureMinDist() ** 2;
  const candidates = reachablePortTiles(game, oceanTile).filter((tile) =>
    blockedTiles.every(
      (blockedTile) =>
        game.euclideanDistSquared(tile, blockedTile) >= minDistSquared,
    ),
  );
  const tile = candidates[index];
  if (tile === undefined) {
    throw new Error(
      "Expected a reachable port tile for offshore oil rig tests",
    );
  }

  player.conquer(tile);
  return player.buildUnit(UnitType.Port, tile, {});
}

function ticksUntilNextCargoLaunch(game: Game): number {
  const remainder = game.ticks() % 100;
  return remainder === 0 ? 100 : 100 - remainder;
}

function firstTradeShip(player: Player) {
  return player.units(UnitType.TradeShip)[0];
}

function nearbyOceanTile(game: Game, tile: TileRef): TileRef {
  for (const neighbor of game.neighbors(tile)) {
    if (game.isOcean(neighbor)) {
      return neighbor;
    }
  }
  throw new Error("Expected an adjacent ocean tile");
}

function nearbyNonOilOceanTile(
  game: Game,
  searchRadius = 3,
): { oilTile: TileRef; nearbyTile: TileRef } | null {
  for (const field of game.oilFields()) {
    for (const oilTile of field.tiles) {
      if (!game.isOcean(oilTile)) {
        continue;
      }

      const originX = game.x(oilTile);
      const originY = game.y(oilTile);
      for (let dy = -searchRadius; dy <= searchRadius; dy++) {
        for (let dx = -searchRadius; dx <= searchRadius; dx++) {
          const distSquared = dx ** 2 + dy ** 2;
          if (distSquared === 0 || distSquared > searchRadius ** 2) {
            continue;
          }

          const x = originX + dx;
          const y = originY + dy;
          if (!game.isValidCoord(x, y)) {
            continue;
          }

          const nearbyTile = game.ref(x, y);
          if (game.isOcean(nearbyTile) && game.oilFieldAt(nearbyTile) === null) {
            return { oilTile, nearbyTile };
          }
        }
      }
    }
  }

  return null;
}

function runUntil(game: Game, predicate: () => boolean, maxTicks = 200): void {
  for (let i = 0; i < maxTicks; i++) {
    if (predicate()) {
      return;
    }
    game.executeNextTick();
  }

  throw new Error("Condition was not reached before maxTicks elapsed");
}

describe("Oil fields", () => {
  test("generates deterministic fixed-footprint fields for the same map", async () => {
    const gameA = await setup("half_land_half_ocean");
    const gameB = await setup("half_land_half_ocean");

    const normalize = (game: Game) =>
      game.oilFields().map((field) => ({
        id: field.id,
        center: field.center,
        maxReserve: field.maxReserve,
        tiles: [...field.tiles],
      }));

    expect(normalize(gameA)).toEqual(normalize(gameB));
  });

  test("generates offshore-only oil fields on ocean tiles", async () => {
    const game = await setup("half_land_half_ocean");

    expect(game.oilFields().length).toBeGreaterThan(0);
    expect(
      game.oilFields().every(
        (field) =>
          game.isOcean(field.center) &&
          field.tiles.length > 0 &&
          field.tiles.every((tile) => game.isOcean(tile)),
      ),
    ).toBe(true);
  });

  test("rejects land OilRig builds even on owned land tiles", async () => {
    const {
      game,
      players: [player],
    } = await createOilGame("half_land_half_ocean", { instantBuild: true });
    const landTile = firstLandTile(game);

    player.conquer(landTile);

    const buildable = player.buildableUnits(landTile, [UnitType.OilRig])[0];
    expect(buildable?.canBuild).toBe(false);
    expect(player.canBuild(UnitType.OilRig, landTile)).toBe(false);
  });

  test("requires a reachable owned port before offshore OilRig placement is allowed", async () => {
    const {
      game,
      players: [player],
    } = await createOilGame("half_land_half_ocean", { instantBuild: true });
    const oceanTile = firstActiveOceanOilTile(game);

    expect(player.canBuild(UnitType.OilRig, oceanTile)).toBe(false);

    buildReachablePort(game, player, oceanTile);

    expect(player.canBuild(UnitType.OilRig, oceanTile)).toBe(oceanTile);
  });

  test("does not snap offshore OilRig placement from nearby water", async () => {
    const {
      game,
      players: [player],
    } = await createOilGame("world", { instantBuild: true });
    const candidate = nearbyNonOilOceanTile(game);

    if (candidate === null) {
      throw new Error("Expected a nearby non-oil ocean tile");
    }

    buildReachablePort(game, player, candidate.oilTile);

    const buildTarget = player.canBuild(UnitType.OilRig, candidate.nearbyTile);

    expect(buildTarget).toBe(false);
  });

  test("rejects blocked offshore OilRig placement even with a reachable port", async () => {
    const {
      game,
      players: [player],
    } = await createOilGame("half_land_half_ocean", { instantBuild: true });
    const oceanTile = firstActiveOceanOilTile(game);

    buildReachablePort(game, player, oceanTile);
    expect(player.canBuild(UnitType.OilRig, oceanTile)).toBe(oceanTile);

    player.buildUnit(UnitType.OilRig, oceanTile, {});

    expect(player.canBuild(UnitType.OilRig, oceanTile)).toBe(false);
  });

  test("deploys an OilRigShip to ocean oil and converts it into an offshore rig", async () => {
    const {
      game,
      players: [player],
    } = await createOilGame("half_land_half_ocean");
    const oceanTile = firstActiveOceanOilTile(game);

    buildReachablePort(game, player, oceanTile);

    game.addExecution(
      new ConstructionExecution(player, UnitType.OilRig, oceanTile),
    );
    runUntil(game, () => player.units(UnitType.OilRigShip).length === 1);

    const ship = player.units(UnitType.OilRigShip)[0];
    expect(ship).toBeDefined();
    expect(ship?.targetTile()).toBe(oceanTile);

    runUntil(game, () =>
      player.units(UnitType.OilRig).some((rig) => rig.tile() === oceanTile),
    );

    const rig = player
      .units(UnitType.OilRig)
      .find((candidate) => candidate.tile() === oceanTile);
    expect(rig).toBeDefined();
    expect(player.units(UnitType.OilRigShip)).toHaveLength(0);
    expect(rig?.isUnderConstruction()).toBe(true);

    executeTicks(
      game,
      (game.config().unitInfo(UnitType.OilRig).constructionDuration ?? 0) + 2,
    );

    expect(rig?.isUnderConstruction()).toBe(false);
    expect(game.isOilRigActive(rig!)).toBe(true);
  });

  test("pending offshore deployments reserve spacing for later oil rigs", async () => {
    const {
      game,
      players: [player],
    } = await createOilGame("half_land_half_ocean");
    const oceanTile = firstActiveOceanOilTile(game);
    const nearbyOceanTile = nearbyOceanOilTileWithinSpacing(game, oceanTile);
    if (nearbyOceanTile === null) {
      throw new Error("Expected a nearby offshore oil tile within spacing");
    }

    buildReachablePort(game, player, oceanTile);

    game.addExecution(
      new ConstructionExecution(player, UnitType.OilRig, oceanTile),
    );
    runUntil(game, () => player.units(UnitType.OilRigShip).length === 1);

    expect(player.canBuild(UnitType.OilRig, nearbyOceanTile)).toBe(false);
  });

  test("same-tick conflicting offshore deployments only launch one ship", async () => {
    const {
      game,
      players: [player],
    } = await createOilGame("half_land_half_ocean");
    const oceanTile = firstActiveOceanOilTile(game);
    const nearbyOceanTile = nearbyOceanOilTileWithinSpacing(game, oceanTile);
    if (nearbyOceanTile === null) {
      throw new Error("Expected a nearby offshore oil tile within spacing");
    }

    buildReachablePort(game, player, oceanTile);

    game.addExecution(
      new ConstructionExecution(player, UnitType.OilRig, oceanTile),
      new ConstructionExecution(player, UnitType.OilRig, nearbyOceanTile),
    );

    game.executeNextTick();
    game.executeNextTick();

    expect(player.units(UnitType.OilRigShip)).toHaveLength(1);
  });

  test("manual extraction no longer depletes offshore fields during deployment", async () => {
    const {
      game,
      players: [player],
    } = await createOilGame("half_land_half_ocean", { instantBuild: true });
    const oceanTile = firstActiveOceanOilTile(game);
    const fieldId = game.oilFieldAt(oceanTile)?.id;

    if (fieldId === undefined) {
      throw new Error("Expected oil field for selected offshore oil tile");
    }

    buildReachablePort(game, player, oceanTile);

    game.addExecution(
      new ConstructionExecution(player, UnitType.OilRig, oceanTile),
    );
    runUntil(game, () => player.units(UnitType.OilRigShip).length === 1);

    const reserveBefore = game.oilFieldById(fieldId)?.remainingReserve ?? 0;
    game.extractOil(fieldId, reserveBefore * 10);
    expect(game.oilFieldById(fieldId)?.remainingReserve).toBe(reserveBefore);

    runUntil(game, () =>
      player.units(UnitType.OilRig).some((rig) => rig.tile() === oceanTile),
    );

    const rig = player
      .units(UnitType.OilRig)
      .find((candidate) => candidate.tile() === oceanTile);
    expect(rig).toBeDefined();

    executeTicks(
      game,
      (game.config().unitInfo(UnitType.OilRig).constructionDuration ?? 0) + 2,
    );

    expect(game.oilFieldById(fieldId)?.remainingReserve).toBe(reserveBefore);
    expect(game.isOilRigActive(rig!)).toBe(true);
  });

  test("offshore rigs keep producing without draining reserve", async () => {
    const {
      game,
      players: [player],
    } = await createOilGame("half_land_half_ocean", { instantBuild: true });
    const oceanTile = firstActiveOceanOilTile(game);
    const field = game.oilFieldAt(oceanTile);

    if (!field) {
      throw new Error("Expected oil field for selected oil tile");
    }

    buildReachablePort(game, player, oceanTile);
    const rig = player.buildUnit(UnitType.OilRig, oceanTile, {});
    const reserveBefore = game.oilFieldById(field.id)?.remainingReserve ?? 0;
    const goldBefore = player.gold();

    executeTicks(game, 10);

    const reserveAfterTick = game.oilFieldById(field.id)?.remainingReserve ?? 0;
    expect(player.gold()).toBe(goldBefore);
    expect(reserveAfterTick).toBe(reserveBefore);
    expect(game.isOilRigActive(rig)).toBe(true);
    expect(player.units(UnitType.TradeShip)).toHaveLength(0);
  });

  test("offshore rigs launch buffered cargo ships and pay on delivery", async () => {
    const {
      game,
      players: [player],
    } = await createOilGame("half_land_half_ocean", { instantBuild: true });
    const oceanTile = firstActiveOceanOilTile(game);

    buildReachablePort(game, player, oceanTile);
    player.buildUnit(UnitType.OilRig, oceanTile, {});

    const goldBefore = player.gold();

    executeTicks(game, 9);
    expect(player.gold()).toBe(goldBefore);
    expect(player.units(UnitType.TradeShip)).toHaveLength(0);

    executeTicks(game, 1);
    expect(player.gold()).toBe(goldBefore);

    executeTicks(game, ticksUntilNextCargoLaunch(game));
    runUntil(game, () => player.units(UnitType.TradeShip).length === 1, 5);

    runUntil(game, () => player.gold() > goldBefore, 300);
    expect(player.gold()).toBeGreaterThan(goldBefore);
  });

  test("offshore rigs become inert without a reachable port and stop generating gold", async () => {
    const {
      game,
      players: [player],
    } = await createOilGame("half_land_half_ocean", { instantBuild: true });
    const oceanTile = firstActiveOceanOilTile(game);
    const fieldId = game.oilFieldAt(oceanTile)?.id;

    if (fieldId === undefined) {
      throw new Error("Expected oil field for selected offshore oil tile");
    }

    const port = buildReachablePort(game, player, oceanTile);
    const rig = player.buildUnit(UnitType.OilRig, oceanTile, {});

    executeTicks(game, 10);

    port.delete(false);
    const reserveBefore = game.oilFieldById(fieldId)?.remainingReserve ?? 0;
    const goldBefore = player.gold();

    expect(game.isOilRigActive(rig)).toBe(false);

    executeTicks(game, ticksUntilNextCargoLaunch(game) + 20);

    expect(game.oilFieldById(fieldId)?.remainingReserve).toBe(reserveBefore);
    expect(player.gold()).toBe(goldBefore);
    expect(player.units(UnitType.TradeShip)).toHaveLength(0);
  });

  test("offshore rigs stop during a port outage and resume cargo launches once service returns", async () => {
    const {
      game,
      players: [player],
    } = await createOilGame("half_land_half_ocean", { instantBuild: true });
    const oceanTile = firstActiveOceanOilTile(game);
    const fieldId = game.oilFieldAt(oceanTile)?.id;

    if (fieldId === undefined) {
      throw new Error("Expected oil field for selected offshore oil tile");
    }

    const port = buildReachablePort(game, player, oceanTile);
    player.buildUnit(UnitType.OilRig, oceanTile, {});

    executeTicks(game, 10);
    const reserveAtOutage = game.oilFieldById(fieldId)?.remainingReserve ?? 0;

    port.delete(false);
    const goldBeforeOutage = player.gold();
    executeTicks(game, ticksUntilNextCargoLaunch(game) + 20);

    expect(player.units(UnitType.TradeShip)).toHaveLength(0);
    expect(game.oilFieldById(fieldId)?.remainingReserve).toBe(reserveAtOutage);
    expect(player.gold()).toBe(goldBeforeOutage);

    player.buildUnit(UnitType.Port, port.tile(), {});
    const goldBeforeRecovery = player.gold();

    executeTicks(game, ticksUntilNextCargoLaunch(game));
    runUntil(game, () => player.units(UnitType.TradeShip).length === 1, 5);
    runUntil(game, () => player.gold() > goldBeforeRecovery, 300);

    expect(player.gold()).toBeGreaterThan(goldBeforeRecovery);
  });

  test("offshore rigs keep paying while any reachable owned port remains", async () => {
    const {
      game,
      players: [player],
    } = await createOilGame("half_land_half_ocean", { instantBuild: true });
    const oceanTile = firstActiveOceanOilTile(game);
    const portTiles = reachablePortTiles(game, oceanTile);

    const nearPort = buildReachablePort(game, player, oceanTile, 0);
    const farPort = buildReachablePort(
      game,
      player,
      oceanTile,
      portTiles.length - 1,
    );
    player.buildUnit(UnitType.OilRig, oceanTile, {});

    const goldBeforeFallback = player.gold();
    executeTicks(game, ticksUntilNextCargoLaunch(game));
    runUntil(game, () => player.units(UnitType.TradeShip).length === 1, 5);

    const ship = firstTradeShip(player);
    expect(ship?.targetUnit()).toBe(nearPort);

    nearPort.delete(false);
    runUntil(game, () => firstTradeShip(player)?.targetUnit() === farPort, 20);
    runUntil(game, () => player.gold() > goldBeforeFallback, 300);

    expect(player.gold()).toBeGreaterThan(goldBeforeFallback);
    expect(farPort.isActive()).toBe(true);
    expect(player.units(UnitType.TradeShip).length).toBeLessThanOrEqual(1);
  });

  test("offshore rigs keep at most one cargo ship in flight per rig", async () => {
    const {
      game,
      players: [player],
    } = await createOilGame("half_land_half_ocean", { instantBuild: true });
    const oceanTile = firstActiveOceanOilTile(game);
    const portTiles = reachablePortTiles(game, oceanTile);

    buildReachablePort(game, player, oceanTile, portTiles.length - 1);
    player.buildUnit(UnitType.OilRig, oceanTile, {});

    let maxTradeShips = 0;
    for (let i = 0; i < 350; i++) {
      game.executeNextTick();
      maxTradeShips = Math.max(
        maxTradeShips,
        player.units(UnitType.TradeShip).length,
      );
    }

    expect(maxTradeShips).toBe(1);
    expect(player.gold()).toBeGreaterThan(0n);
  });

  test("offshore rigs pay at a much lower steady rate", async () => {
    const {
      game,
      players: [player],
    } = await createOilGame("half_land_half_ocean", { instantBuild: true });
    const oceanTile = firstActiveOceanOilTile(game);

    buildReachablePort(game, player, oceanTile);
    player.buildUnit(UnitType.OilRig, oceanTile, {});

    const goldBefore = player.gold();

    runUntil(game, () => player.units(UnitType.TradeShip).length === 1, 150);
    runUntil(game, () => player.gold() > goldBefore, 300);

    const earned = player.gold() - goldBefore;

    expect(earned).toBeGreaterThan(20_000n);
    expect(earned).toBeLessThan(40_000n);
  });

  test("transport ships can capture hostile finished offshore rigs in place", async () => {
    const {
      game,
      players: [attacker, defender],
    } = await createOilGame("half_land_half_ocean", { instantBuild: true }, [
      new PlayerInfo("attacker", PlayerType.Human, null, "attacker_id"),
      new PlayerInfo("defender", PlayerType.Human, null, "defender_id"),
    ]);
    const oceanTile = firstActiveOceanOilTile(game);

    buildReachablePort(game, attacker, oceanTile, 0);
    const rig = defender.buildUnit(UnitType.OilRig, oceanTile, {});
    executeTicks(game, 20);

    expect(attacker.canBuild(UnitType.TransportShip, oceanTile)).not.toBe(
      false,
    );

    game.addExecution(new TransportShipExecution(attacker, oceanTile, 100));
    runUntil(game, () => rig.owner() === attacker, 500);

    expect(rig.owner()).toBe(attacker);
    expect(game.owner(oceanTile).isPlayer()).toBe(false);
    expect(attacker.units(UnitType.TransportShip)).toHaveLength(0);
  });

  test("transport attacks cannot target empty ocean or friendly offshore rigs", async () => {
    const {
      game,
      players: [attacker, defender],
    } = await createOilGame("half_land_half_ocean", { instantBuild: true }, [
      new PlayerInfo("attacker", PlayerType.Human, null, "attacker_id"),
      new PlayerInfo("defender", PlayerType.Human, null, "defender_id"),
    ]);
    const oceanTile = firstActiveOceanOilTile(game);
    const emptyOceanTile = nearbyOceanTile(game, oceanTile);

    buildReachablePort(game, attacker, oceanTile, 0);
    executeTicks(game, 20);

    expect(attacker.canBuild(UnitType.TransportShip, emptyOceanTile)).toBe(
      false,
    );

    const friendlyRig = attacker.buildUnit(UnitType.OilRig, oceanTile, {});
    expect(attacker.canBuild(UnitType.TransportShip, oceanTile)).toBe(false);

    friendlyRig.delete(false);
    defender.buildUnit(UnitType.OilRig, oceanTile, {});

    expect(attacker.canBuild(UnitType.TransportShip, oceanTile)).not.toBe(
      false,
    );
  });
});
