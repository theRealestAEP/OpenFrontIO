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
import { GameConfig } from "../../../src/core/Schemas";
import { setup } from "../../util/Setup";
import { executeTicks } from "../../util/utils";

async function createOilGame(
  mapName = "plains",
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

function firstActiveLandOilTile(game: Game): TileRef {
  for (const field of game.oilFields()) {
    if (field.remainingReserve <= 0) {
      continue;
    }
    const tile = field.tiles.find((candidate) => game.isLand(candidate));
    if (tile !== undefined) {
      return tile;
    }
  }
  throw new Error("Expected at least one land oil tile on the test map");
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

function firstMixedOilFieldTiles(game: Game): {
  fieldId: number;
  landTile: TileRef;
  oceanTile: TileRef;
} {
  for (const field of game.oilFields()) {
    if (field.remainingReserve <= 0) {
      continue;
    }
    const landTile = field.tiles.find((tile) => game.isLand(tile));
    const oceanTile = field.tiles.find((tile) => game.isOcean(tile));
    if (landTile !== undefined && oceanTile !== undefined) {
      return {
        fieldId: field.id,
        landTile,
        oceanTile,
      };
    }
  }
  throw new Error("Expected at least one mixed land/ocean oil field");
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
) {
  const candidates = reachablePortTiles(game, oceanTile);
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

function ticksUntilNextOilTick(game: Game): number {
  const remainder = game.ticks() % 10;
  return remainder === 0 ? 10 : 10 - remainder;
}

function runUntilWithTicks(
  game: Game,
  predicate: () => boolean,
  maxTicks = 200,
): number {
  let elapsed = 0;
  for (; elapsed < maxTicks; elapsed++) {
    if (predicate()) {
      return elapsed;
    }
    game.executeNextTick();
  }

  throw new Error("Condition was not reached before maxTicks elapsed");
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
    const gameA = await setup("plains");
    const gameB = await setup("plains");

    const normalize = (game: Game) =>
      game.oilFields().map((field) => ({
        id: field.id,
        center: field.center,
        maxReserve: field.maxReserve,
        tiles: [...field.tiles],
      }));

    expect(normalize(gameA)).toEqual(normalize(gameB));
  });

  test("extends deterministic mixed oil fields into ocean tiles", async () => {
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
    expect(
      gameA.oilFields().some((field) => {
        const hasLand = field.tiles.some((tile) => gameA.isLand(tile));
        const hasOcean = field.tiles.some((tile) => gameA.isOcean(tile));
        return hasLand && hasOcean;
      }),
    ).toBe(true);
  });

  test("allows land OilRig builds only on owned fields with remaining oil", async () => {
    const {
      game,
      players: [player],
    } = await createOilGame();
    const oilTile = firstActiveLandOilTile(game);

    const unownedBuildable = player.buildableUnits(oilTile, [
      UnitType.OilRig,
    ])[0];
    expect(unownedBuildable?.canBuild).toBe(false);

    player.conquer(oilTile);

    const activeBuildable = player.buildableUnits(oilTile, [
      UnitType.OilRig,
    ])[0];
    expect(activeBuildable?.canBuild).not.toBe(false);

    for (const overlappingField of overlappingFieldsAt(game, oilTile)) {
      game.extractOil(overlappingField.id, overlappingField.remainingReserve);
    }

    const depletedBuildable = player.buildableUnits(oilTile, [
      UnitType.OilRig,
    ])[0];
    expect(depletedBuildable?.canBuild).toBe(false);
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

  test("still finishes offshore construction as a dormant rig if the field depletes during travel", async () => {
    const {
      game,
      players: [player],
    } = await createOilGame("half_land_half_ocean", { instantBuild: true });
    const oceanTile = firstActiveOceanOilTile(game);

    buildReachablePort(game, player, oceanTile);

    game.addExecution(
      new ConstructionExecution(player, UnitType.OilRig, oceanTile),
    );
    runUntil(game, () => player.units(UnitType.OilRigShip).length === 1);

    for (const overlappingField of overlappingFieldsAt(game, oceanTile)) {
      game.extractOil(overlappingField.id, overlappingField.remainingReserve);
    }

    runUntil(game, () =>
      player.units(UnitType.OilRig).some((rig) => rig.tile() === oceanTile),
    );

    const rig = player
      .units(UnitType.OilRig)
      .find((candidate) => candidate.tile() === oceanTile);
    expect(rig).toBeDefined();
    expect(game.isOilRigActive(rig!)).toBe(false);
  });

  test("pays passive gold while reserves remain and goes dormant after depletion", async () => {
    const {
      game,
      players: [player],
    } = await createOilGame();
    const oilTile = firstActiveLandOilTile(game);
    const field = game.oilFieldAt(oilTile);

    if (!field) {
      throw new Error("Expected oil field for selected oil tile");
    }

    player.conquer(oilTile);
    const rig = player.buildUnit(UnitType.OilRig, oilTile, {});

    const goldBefore = player.gold();
    const reserveBefore = game.oilFieldById(field.id)?.remainingReserve ?? 0;

    for (let i = 0; i < 12; i++) {
      game.executeNextTick();
    }

    const reserveAfterTick = game.oilFieldById(field.id)?.remainingReserve ?? 0;
    expect(player.gold()).toBeGreaterThan(goldBefore);
    expect(reserveAfterTick).toBeLessThan(reserveBefore);
    expect(game.isOilRigActive(rig)).toBe(true);

    for (const overlappingField of overlappingFieldsAt(game, oilTile)) {
      const remaining =
        game.oilFieldById(overlappingField.id)?.remainingReserve ?? 0;
      game.extractOil(overlappingField.id, remaining);
    }
    game.executeNextTick();

    const goldAfterDepletion = player.gold();
    expect(game.oilFieldById(field.id)?.remainingReserve).toBe(0);
    expect(game.isOilRigActive(rig)).toBe(false);

    game.executeNextTick();
    expect(player.gold()).toBe(goldAfterDepletion);
  });

  test("offshore rigs launch cargo ships on a 100 tick interval and pay on delivery", async () => {
    const {
      game,
      players: [player],
    } = await createOilGame("half_land_half_ocean", { instantBuild: true });
    const oceanTile = firstActiveOceanOilTile(game);

    buildReachablePort(game, player, oceanTile);
    player.buildUnit(UnitType.OilRig, oceanTile, {});

    const goldBefore = player.gold();
    const ticksBeforeLaunch = ticksUntilNextCargoLaunch(game);

    executeTicks(game, Math.max(0, ticksBeforeLaunch - 1));
    expect(player.units(UnitType.TradeShip)).toHaveLength(0);
    expect(player.gold()).toBe(goldBefore);

    executeTicks(game, 1);
    runUntil(game, () => player.units(UnitType.TradeShip).length === 1, 5);

    runUntil(game, () => player.gold() > goldBefore, 300);
    expect(player.gold()).toBeGreaterThan(goldBefore);
  });

  test("offshore rigs become inert without a reachable port and stop draining reserve", async () => {
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

    executeTicks(game, 20);

    expect(game.oilFieldById(fieldId)?.remainingReserve).toBe(reserveBefore);
    expect(player.gold()).toBe(goldBefore);
    expect(player.units(UnitType.TradeShip)).toHaveLength(0);
  });

  test("buffered offshore cargo survives a port outage and launches once service returns", async () => {
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

    executeTicks(game, 20);
    const reserveAtOutage = game.oilFieldById(fieldId)?.remainingReserve ?? 0;

    port.delete(false);
    executeTicks(game, Math.max(0, ticksUntilNextCargoLaunch(game) - 1));

    expect(player.units(UnitType.TradeShip)).toHaveLength(0);
    expect(game.oilFieldById(fieldId)?.remainingReserve).toBe(reserveAtOutage);

    player.buildUnit(UnitType.Port, port.tile(), {});
    const goldBeforeDelivery = player.gold();

    executeTicks(game, 1);
    runUntil(game, () => player.units(UnitType.TradeShip).length === 1, 5);
    runUntil(game, () => player.gold() > goldBeforeDelivery, 300);

    expect(player.gold() - goldBeforeDelivery).toBeGreaterThan(50_000n);
  });

  test("offshore cargo ships choose the nearest reachable owned port and reroute if needed", async () => {
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

    executeTicks(game, ticksUntilNextCargoLaunch(game));
    runUntil(game, () => player.units(UnitType.TradeShip).length === 1, 5);

    const ship = firstTradeShip(player);
    expect(ship?.targetUnit()).toBe(nearPort);

    nearPort.delete(false);
    runUntil(game, () => firstTradeShip(player)?.targetUnit() === farPort, 10);

    expect(firstTradeShip(player)?.targetUnit()).toBe(farPort);
  });

  test("offshore cargo ships are deleted if no valid owned port remains during transit", async () => {
    const {
      game,
      players: [player],
    } = await createOilGame("half_land_half_ocean", { instantBuild: true });
    const oceanTile = firstActiveOceanOilTile(game);

    const nearPort = buildReachablePort(game, player, oceanTile, 0);
    const farPort = buildReachablePort(
      game,
      player,
      oceanTile,
      reachablePortTiles(game, oceanTile).length - 1,
    );
    player.buildUnit(UnitType.OilRig, oceanTile, {});

    executeTicks(game, ticksUntilNextCargoLaunch(game));
    runUntil(game, () => player.units(UnitType.TradeShip).length === 1, 5);

    const goldBefore = player.gold();
    nearPort.delete(false);
    farPort.delete(false);

    runUntil(game, () => player.units(UnitType.TradeShip).length === 0, 20);

    expect(player.gold()).toBe(goldBefore);
  });

  test("offshore rigs drain and earn faster than equivalent land rigs", async () => {
    const {
      game: landGame,
      players: [landPlayer],
    } = await createOilGame("half_land_half_ocean", { instantBuild: true });
    const {
      game: oceanGame,
      players: [oceanPlayer],
    } = await createOilGame("half_land_half_ocean", { instantBuild: true });
    const { fieldId, landTile, oceanTile } = firstMixedOilFieldTiles(landGame);

    landPlayer.conquer(landTile);
    landPlayer.buildUnit(UnitType.OilRig, landTile, {});

    buildReachablePort(oceanGame, oceanPlayer, oceanTile);
    oceanPlayer.buildUnit(UnitType.OilRig, oceanTile, {});

    const landGoldBefore = landPlayer.gold();
    const oceanGoldBefore = oceanPlayer.gold();
    const landReserveBefore =
      landGame.oilFieldById(fieldId)?.remainingReserve ?? 0;
    const oceanReserveBefore =
      oceanGame.oilFieldById(fieldId)?.remainingReserve ?? 0;

    executeTicks(landGame, 10);
    executeTicks(oceanGame, 10);

    const landReserveAfter =
      landGame.oilFieldById(fieldId)?.remainingReserve ?? 0;
    const oceanReserveAfter =
      oceanGame.oilFieldById(fieldId)?.remainingReserve ?? 0;

    expect(oceanReserveBefore - oceanReserveAfter).toBeGreaterThan(
      landReserveBefore - landReserveAfter,
    );

    const extraTicks = runUntilWithTicks(
      oceanGame,
      () => oceanPlayer.gold() > oceanGoldBefore,
      400,
    );
    executeTicks(landGame, extraTicks);

    expect(oceanPlayer.gold() - oceanGoldBefore).toBeGreaterThan(
      landPlayer.gold() - landGoldBefore,
    );
  });

  test("land and offshore rigs on the same field share reserves and one owner bucket", async () => {
    const {
      game: sharedGame,
      players: [sharedPlayer],
    } = await createOilGame("half_land_half_ocean", { instantBuild: true });
    const {
      game: landOnlyGame,
      players: [landOnlyPlayer],
    } = await createOilGame("half_land_half_ocean", { instantBuild: true });
    const {
      game: offshoreOnlyGame,
      players: [offshoreOnlyPlayer],
    } = await createOilGame("half_land_half_ocean", { instantBuild: true });

    const sharedTiles = firstMixedOilFieldTiles(sharedGame);
    const landOnlyTiles = firstMixedOilFieldTiles(landOnlyGame);
    const offshoreOnlyTiles = firstMixedOilFieldTiles(offshoreOnlyGame);

    sharedPlayer.conquer(sharedTiles.landTile);
    sharedPlayer.buildUnit(UnitType.OilRig, sharedTiles.landTile, {});
    buildReachablePort(sharedGame, sharedPlayer, sharedTiles.oceanTile);
    sharedPlayer.buildUnit(UnitType.OilRig, sharedTiles.oceanTile, {});

    landOnlyPlayer.conquer(landOnlyTiles.landTile);
    landOnlyPlayer.buildUnit(UnitType.OilRig, landOnlyTiles.landTile, {});

    buildReachablePort(
      offshoreOnlyGame,
      offshoreOnlyPlayer,
      offshoreOnlyTiles.oceanTile,
    );
    offshoreOnlyPlayer.buildUnit(
      UnitType.OilRig,
      offshoreOnlyTiles.oceanTile,
      {},
    );

    const sharedFieldId = sharedGame.oilFieldAt(sharedTiles.landTile)?.id;
    const landOnlyFieldId = landOnlyGame.oilFieldAt(landOnlyTiles.landTile)?.id;
    const offshoreOnlyFieldId = offshoreOnlyGame.oilFieldAt(
      offshoreOnlyTiles.oceanTile,
    )?.id;

    if (
      sharedFieldId === undefined ||
      landOnlyFieldId === undefined ||
      offshoreOnlyFieldId === undefined
    ) {
      throw new Error("Expected active oil fields for shared reserve test");
    }

    const sharedReserveBefore =
      sharedGame.oilFieldById(sharedFieldId)?.remainingReserve ?? 0;
    const landOnlyReserveBefore =
      landOnlyGame.oilFieldById(landOnlyFieldId)?.remainingReserve ?? 0;
    const offshoreOnlyReserveBefore =
      offshoreOnlyGame.oilFieldById(offshoreOnlyFieldId)?.remainingReserve ?? 0;

    const sharedGoldBefore = sharedPlayer.gold();
    executeTicks(sharedGame, 10);
    executeTicks(landOnlyGame, 10);
    executeTicks(offshoreOnlyGame, 10);

    const sharedDrain =
      sharedReserveBefore -
      (sharedGame.oilFieldById(sharedFieldId)?.remainingReserve ?? 0);
    const landOnlyDrain =
      landOnlyReserveBefore -
      (landOnlyGame.oilFieldById(landOnlyFieldId)?.remainingReserve ?? 0);
    const offshoreOnlyDrain =
      offshoreOnlyReserveBefore -
      (offshoreOnlyGame.oilFieldById(offshoreOnlyFieldId)?.remainingReserve ??
        0);

    expect(sharedDrain).toBeGreaterThan(0);
    expect(sharedDrain).toBeLessThan(landOnlyDrain + offshoreOnlyDrain);
    expect(sharedPlayer.gold()).toBeGreaterThan(sharedGoldBefore);
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
