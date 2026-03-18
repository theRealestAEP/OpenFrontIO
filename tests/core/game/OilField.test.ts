import {
  Game,
  Player,
  PlayerInfo,
  PlayerType,
  UnitType,
} from "../../../src/core/game/Game";
import { TileRef } from "../../../src/core/game/GameMap";
import { OilExecution } from "../../../src/core/execution/OilExecution";
import { setup } from "../../util/Setup";

async function createOilGame(): Promise<{ game: Game; player: Player }> {
  const game = await setup(
    "plains",
    {
      instantBuild: true,
    },
    [new PlayerInfo("player", PlayerType.Human, null, "player_id")],
  );

  while (game.inSpawnPhase()) {
    game.executeNextTick();
  }

  game.addExecution(new OilExecution());

  const player = game.player("player_id");
  player.addGold(10_000_000n);

  return { game, player };
}

function firstActiveOilTile(game: Game): TileRef {
  const field = game
    .oilFields()
    .find((candidate) => candidate.remainingReserve > 0 && candidate.tiles[0]);
  if (!field) {
    throw new Error("Expected at least one oil field on the test map");
  }
  return field.tiles[0];
}

function overlappingFieldsAt(game: Game, tile: TileRef) {
  return game.oilFields().filter((field) => field.tiles.includes(tile));
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

  test("allows OilRig builds only on owned fields with remaining oil", async () => {
    const { game, player } = await createOilGame();
    const oilTile = firstActiveOilTile(game);
    const field = game.oilFieldAt(oilTile);

    if (!field) {
      throw new Error("Expected oil field for selected oil tile");
    }

    const unownedBuildable = player.buildableUnits(oilTile, [UnitType.OilRig])[0];
    expect(unownedBuildable?.canBuild).toBe(false);

    player.conquer(oilTile);

    const activeBuildable = player.buildableUnits(oilTile, [UnitType.OilRig])[0];
    expect(activeBuildable?.canBuild).not.toBe(false);

    for (const overlappingField of overlappingFieldsAt(game, oilTile)) {
      game.extractOil(overlappingField.id, overlappingField.remainingReserve);
    }

    const depletedBuildable = player.buildableUnits(oilTile, [
      UnitType.OilRig,
    ])[0];
    expect(depletedBuildable?.canBuild).toBe(false);
  });

  test("pays passive gold while reserves remain and goes dormant after depletion", async () => {
    const { game, player } = await createOilGame();
    const oilTile = firstActiveOilTile(game);
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
});
