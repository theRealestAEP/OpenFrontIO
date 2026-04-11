import { MoveWarshipExecution } from "../src/core/execution/MoveWarshipExecution";
import { OilExecution } from "../src/core/execution/OilExecution";
import { WarshipExecution } from "../src/core/execution/WarshipExecution";
import {
  Game,
  Player,
  PlayerInfo,
  PlayerType,
  UnitType,
} from "../src/core/game/Game";
import { setup } from "./util/Setup";
import { executeTicks } from "./util/utils";

const coastX = 7;
let game: Game;
let player1: Player;
let player2: Player;

function firstActiveOceanOilTile(game: Game): number {
  for (const field of game.oilFields()) {
    if (field.remainingReserve <= 0) {
      continue;
    }
    const tile = field.tiles.find((candidate) => game.isOcean(candidate));
    if (tile !== undefined) {
      return tile;
    }
  }
  throw new Error("Expected an ocean oil tile for warship tests");
}

function nearbyOceanTile(game: Game, tile: number): number {
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

function ticksUntilNextCargoLaunch(game: Game): number {
  const remainder = game.ticks() % 100;
  return remainder === 0 ? 100 : 100 - remainder;
}

function buildReachablePort(game: Game, player: Player, oceanTile: number) {
  const minDistSquared = game.config().structureMinDist() ** 2;

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
      if (
        game.nearbyUnits(tile, 0, UnitType.Port, undefined, true).length > 0
      ) {
        continue;
      }
      player.conquer(tile);
      return player.buildUnit(UnitType.Port, tile, {});
    }
  }

  throw new Error("Expected a reachable port tile");
}

describe("Warship", () => {
  beforeEach(async () => {
    game = await setup(
      "half_land_half_ocean",
      {
        infiniteGold: true,
        instantBuild: true,
      },
      [
        new PlayerInfo("boat dude", PlayerType.Human, null, "player_1_id"),
        new PlayerInfo("boat dude", PlayerType.Human, null, "player_2_id"),
      ],
    );

    while (game.inSpawnPhase()) {
      game.executeNextTick();
    }

    player1 = game.player("player_1_id");
    player2 = game.player("player_2_id");
  });

  test("Warship heals only if player has port", async () => {
    const maxHealth = game.config().unitInfo(UnitType.Warship).maxHealth;
    if (typeof maxHealth !== "number") {
      expect(typeof maxHealth).toBe("number");
      throw new Error("unreachable");
    }

    const port = player1.buildUnit(UnitType.Port, game.ref(coastX, 10), {});
    const warship = player1.buildUnit(
      UnitType.Warship,
      game.ref(coastX + 1, 10),
      {
        patrolTile: game.ref(coastX + 1, 10),
      },
    );
    game.addExecution(new WarshipExecution(warship));

    game.executeNextTick();

    expect(warship.health()).toBe(maxHealth);
    warship.modifyHealth(-10);
    expect(warship.health()).toBe(maxHealth - 10);
    game.executeNextTick();
    expect(warship.health()).toBe(maxHealth - 9);

    port.delete();

    game.executeNextTick();
    expect(warship.health()).toBe(maxHealth - 9);
  });

  test("Warship captures trade if player has port", async () => {
    const portTile = game.ref(coastX, 10);
    player1.buildUnit(UnitType.Port, portTile, {});
    game.addExecution(
      new WarshipExecution(
        player1.buildUnit(UnitType.Warship, portTile, {
          patrolTile: portTile,
        }),
      ),
    );

    const tradeShip = player2.buildUnit(
      UnitType.TradeShip,
      game.ref(coastX + 1, 7),
      {
        targetUnit: player2.buildUnit(UnitType.Port, game.ref(coastX, 10), {}),
      },
    );

    expect(tradeShip.owner().id()).toBe(player2.id());
    // Let plenty of time for A* to execute
    for (let i = 0; i < 10; i++) {
      game.executeNextTick();
    }
    expect(tradeShip.owner()).toBe(player1);
  });

  test("Warship do not capture trade if player has no port", async () => {
    game.addExecution(
      new WarshipExecution(
        player1.buildUnit(UnitType.Warship, game.ref(coastX + 1, 11), {
          patrolTile: game.ref(coastX + 1, 11),
        }),
      ),
    );

    const tradeShip = player2.buildUnit(
      UnitType.TradeShip,
      game.ref(coastX + 1, 11),
      {
        targetUnit: player1.buildUnit(UnitType.Port, game.ref(coastX, 11), {}),
      },
    );

    expect(tradeShip.owner().id()).toBe(player2.id());
    // Let plenty of time for warship to potentially capture trade ship
    for (let i = 0; i < 10; i++) {
      game.executeNextTick();
    }

    expect(tradeShip.owner().id()).toBe(player2.id());
  });

  test("Warship does not target trade ships that are safe from pirates", async () => {
    // build port so warship can target trade ships
    player1.buildUnit(UnitType.Port, game.ref(coastX, 10), {});

    const warship = player1.buildUnit(
      UnitType.Warship,
      game.ref(coastX + 1, 10),
      {
        patrolTile: game.ref(coastX + 1, 10),
      },
    );
    game.addExecution(new WarshipExecution(warship));

    const tradeShip = player2.buildUnit(
      UnitType.TradeShip,
      game.ref(coastX + 1, 10),
      {
        targetUnit: player2.buildUnit(UnitType.Port, game.ref(coastX, 10), {}),
      },
    );

    tradeShip.setSafeFromPirates();

    executeTicks(game, 10);

    expect(tradeShip.owner().id()).toBe(player2.id());
  });

  test("Warship moves to new patrol tile", async () => {
    game.config().warshipTargettingRange = () => 1;

    const warship = player1.buildUnit(
      UnitType.Warship,
      game.ref(coastX + 1, 10),
      {
        patrolTile: game.ref(coastX + 1, 10),
      },
    );

    game.addExecution(new WarshipExecution(warship));

    game.addExecution(
      new MoveWarshipExecution(player1, warship.id(), game.ref(coastX + 5, 15)),
    );

    executeTicks(game, 10);

    expect(warship.patrolTile()).toBe(game.ref(coastX + 5, 15));
  });

  test("Warship does not not target trade ships outside of patrol range", async () => {
    game.config().warshipTargettingRange = () => 3;

    // build port so warship can target trade ships
    player1.buildUnit(UnitType.Port, game.ref(coastX, 10), {});

    const warship = player1.buildUnit(
      UnitType.Warship,
      game.ref(coastX + 1, 10),
      {
        patrolTile: game.ref(coastX + 1, 10),
      },
    );
    game.addExecution(new WarshipExecution(warship));

    const tradeShip = player2.buildUnit(
      UnitType.TradeShip,
      game.ref(coastX + 1, 15),
      {
        targetUnit: player2.buildUnit(UnitType.Port, game.ref(coastX, 10), {}),
      },
    );

    executeTicks(game, 10);

    // Trade ship should not be captured
    expect(tradeShip.owner().id()).toBe(player2.id());
  });

  test("MoveWarshipExecution fails if player is not the owner", async () => {
    const originalPatrolTile = game.ref(coastX + 1, 10);
    const warship = player1.buildUnit(
      UnitType.Warship,
      game.ref(coastX + 1, 5),
      {
        patrolTile: originalPatrolTile,
      },
    );
    new MoveWarshipExecution(
      player2,
      warship.id(),
      game.ref(coastX + 5, 15),
    ).init(game, 0);
    expect(warship.patrolTile()).toBe(originalPatrolTile);
  });

  test("MoveWarshipExecution fails if warship is not active", async () => {
    const originalPatrolTile = game.ref(coastX + 1, 10);
    const warship = player1.buildUnit(
      UnitType.Warship,
      game.ref(coastX + 1, 5),
      {
        patrolTile: originalPatrolTile,
      },
    );
    warship.delete();
    new MoveWarshipExecution(
      player1,
      warship.id(),
      game.ref(coastX + 5, 15),
    ).init(game, 0);
    expect(warship.patrolTile()).toBe(originalPatrolTile);
  });

  test("MoveWarshipExecution fails gracefully if warship not found", async () => {
    const exec = new MoveWarshipExecution(
      player1,
      123,
      game.ref(coastX + 5, 15),
    );

    // Verify that no error is thrown.
    exec.init(game, 0);

    expect(exec.isActive()).toBe(false);
  });

  test("Warship does not capture finished offshore oil rigs", async () => {
    const oceanTile = firstActiveOceanOilTile(game);
    const rig = player2.buildUnit(UnitType.OilRig, oceanTile, {});
    const warshipTile = nearbyOceanTile(game, oceanTile);

    game.addExecution(
      new WarshipExecution(
        player1.buildUnit(UnitType.Warship, warshipTile, {
          patrolTile: warshipTile,
        }),
      ),
    );

    executeTicks(game, 10);

    expect(rig.owner()).toBe(player2);
  });

  test("Warship does not capture offshore oil rigs while they are under construction", async () => {
    const oceanTile = firstActiveOceanOilTile(game);
    const rig = player2.buildUnit(UnitType.OilRig, oceanTile, {});
    rig.setUnderConstruction(true);
    const warshipTile = nearbyOceanTile(game, oceanTile);

    game.addExecution(
      new WarshipExecution(
        player1.buildUnit(UnitType.Warship, warshipTile, {
          patrolTile: warshipTile,
        }),
      ),
    );

    executeTicks(game, 10);

    expect(rig.owner()).toBe(player2);
  });

  test("Warship does not capture offshore deploy ships in v1", async () => {
    const oceanTile = firstActiveOceanOilTile(game);
    const shipTile = nearbyOceanTile(game, oceanTile);
    const deployShip = player2.buildUnit(UnitType.OilRigShip, shipTile, {
      targetTile: oceanTile,
    });
    const warshipTile = nearbyOceanTile(game, shipTile);

    game.addExecution(
      new WarshipExecution(
        player1.buildUnit(UnitType.Warship, warshipTile, {
          patrolTile: warshipTile,
        }),
      ),
    );

    executeTicks(game, 10);

    expect(deployShip.owner()).toBe(player2);
  });

  test("Warship captures offshore oil cargo ships and pays out on delivery", async () => {
    game.addExecution(new OilExecution());

    const oceanTile = firstActiveOceanOilTile(game);
    player1.buildUnit(UnitType.Port, game.ref(coastX, 12), {});
    buildReachablePort(game, player2, oceanTile);
    player2.buildUnit(UnitType.OilRig, oceanTile, {});
    const warshipTile = nearbyOceanTile(game, oceanTile);

    game.addExecution(
      new WarshipExecution(
        player1.buildUnit(UnitType.Warship, warshipTile, {
          patrolTile: warshipTile,
        }),
      ),
    );

    const pirateGoldBefore = player1.gold();
    const ownerGoldBefore = player2.gold();

    executeTicks(game, ticksUntilNextCargoLaunch(game));
    runUntil(game, () => game.unitCount(UnitType.TradeShip) === 1, 5);

    const cargoShip = game.units(UnitType.TradeShip)[0];

    runUntil(game, () => cargoShip.owner() === player1, 100);
    expect(player1.gold()).toBe(pirateGoldBefore);

    runUntil(game, () => player1.gold() > pirateGoldBefore, 300);
    expect(player1.gold()).toBeGreaterThan(pirateGoldBefore);
    expect(player2.gold()).toBe(ownerGoldBefore);
  });
});
