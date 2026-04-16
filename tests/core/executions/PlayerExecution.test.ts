import { PlayerExecution } from "../../../src/core/execution/PlayerExecution";
import {
  Game,
  Player,
  PlayerInfo,
  PlayerType,
  UnitType,
} from "../../../src/core/game/Game";
import { setup } from "../../util/Setup";
import { executeTicks } from "../../util/utils";

let game: Game;
let player: Player;
let otherPlayer: Player;

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
      player.conquer(tile);
      return player.buildUnit(UnitType.Port, tile, {});
    }
  }

  throw new Error("Expected a reachable port tile for offshore rig test");
}

describe("PlayerExecution", () => {
  beforeEach(async () => {
    game = await setup(
      "big_plains",
      {
        infiniteGold: true,
        instantBuild: true,
      },
      [
        new PlayerInfo("player", PlayerType.Human, "client_id1", "player_id"),
        new PlayerInfo("other", PlayerType.Human, "client_id2", "other_id"),
      ],
    );

    while (game.inSpawnPhase()) {
      game.executeNextTick();
    }

    player = game.player("player_id");
    otherPlayer = game.player("other_id");

    game.addExecution(new PlayerExecution(player));
    game.addExecution(new PlayerExecution(otherPlayer));
  });

  test("DefensePost lv. 1 is destroyed when tile owner changes", () => {
    const tile = game.ref(50, 50);
    player.conquer(tile);
    const defensePost = player.buildUnit(UnitType.DefensePost, tile, {});

    game.executeNextTick();
    expect(game.unitCount(UnitType.DefensePost)).toBe(1);
    expect(defensePost.level()).toBe(1);

    otherPlayer.conquer(tile);
    executeTicks(game, 2);

    expect(game.unitCount(UnitType.DefensePost)).toBe(0);
  });

  test("DefensePost lv. 2+ is downgraded when tile owner changes", () => {
    const tile = game.ref(50, 50);
    player.conquer(tile);
    const defensePost = player.buildUnit(UnitType.DefensePost, tile, {});
    defensePost.increaseLevel();

    expect(defensePost.level()).toBe(2);
    expect(game.unitCount(UnitType.DefensePost)).toBe(2); // unitCount sums levels
    expect(player.units(UnitType.DefensePost)).toHaveLength(1);
    expect(defensePost.isActive()).toBe(true);

    otherPlayer.conquer(tile);
    executeTicks(game, 2);

    expect(defensePost.level()).toBe(1);
    expect(game.unitCount(UnitType.DefensePost)).toBe(1);
    expect(otherPlayer.units(UnitType.DefensePost)).toHaveLength(1);
    expect(defensePost.owner()).toBe(otherPlayer);
    expect(defensePost.isActive()).toBe(true);
  });

  test("Non-DefensePost structures are transferred (not downgraded) when tile owner changes", () => {
    const tile = game.ref(50, 50);
    player.conquer(tile);
    const city = player.buildUnit(UnitType.City, tile, {});

    expect(game.unitCount(UnitType.City)).toBe(1);
    expect(city.level()).toBe(1);
    expect(city.owner()).toBe(player);
    expect(city.isActive()).toBe(true);

    otherPlayer.conquer(tile);
    executeTicks(game, 2);

    expect(game.unitCount(UnitType.City)).toBe(1);
    expect(city.level()).toBe(1);
    expect(city.owner()).toBe(otherPlayer);
    expect(city.isActive()).toBe(true);
  });

  test("Offshore oil rigs are not deleted on unowned ocean tiles", async () => {
    const offshoreGame = await setup(
      "half_land_half_ocean",
      {
        infiniteGold: true,
        instantBuild: true,
      },
      [
        new PlayerInfo(
          "offshore",
          PlayerType.Human,
          "client_id3",
          "offshore_id",
        ),
      ],
    );

    while (offshoreGame.inSpawnPhase()) {
      offshoreGame.executeNextTick();
    }

    const offshorePlayer = offshoreGame.player("offshore_id");
    offshoreGame.addExecution(new PlayerExecution(offshorePlayer));

    const oceanTile = offshoreGame
      .oilFields()
      .flatMap((field) => field.tiles)
      .find((tile) => offshoreGame.isOcean(tile));
    if (oceanTile === undefined) {
      throw new Error("Expected ocean oil tile for offshore rig test");
    }

    offshorePlayer.conquer(offshoreGame.ref(0, 0));
    buildReachablePort(offshoreGame, offshorePlayer, oceanTile);
    const rig = offshorePlayer.buildUnit(UnitType.OilRig, oceanTile, {});

    executeTicks(offshoreGame, 3);

    expect(rig.isActive()).toBe(true);
    expect(offshorePlayer.units(UnitType.OilRig)).toHaveLength(1);
  });
});
