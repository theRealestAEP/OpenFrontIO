import { describe, expect, it, vi } from "vitest";
import { GameMode, HumansVsNations } from "../../src/core/game/Game";
import { MapPlaylist } from "../../src/server/MapPlaylist";

vi.mock("../../src/server/Logger", () => ({
  logger: {
    child: () => ({
      info: () => {},
      warn: () => {},
      error: () => {},
    }),
  },
}));

describe("MapPlaylist zombie lane", () => {
  it("rotates between zombie FFA and zombie Team configs", async () => {
    const playlist = new MapPlaylist();

    const first = await playlist.gameConfig("zombie");
    const second = await playlist.gameConfig("zombie");

    expect(first.specialRuleset).toBe("zombie_survival");
    expect(second.specialRuleset).toBe("zombie_survival");
    expect(first.nations).toBe("default");
    expect(second.nations).toBe("default");
    expect(first.gameMode).toBe(GameMode.FFA);
    expect(second.gameMode).toBe(GameMode.Team);
    expect(second.playerTeams).not.toBe(HumansVsNations);
  });
});
