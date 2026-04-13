vi.mock("../../../src/client/Utils", () => ({
  translateText: vi.fn((key: string) => key),
}));

import {
  GameConfigSettings,
  GameConfigSettingsData,
} from "../../../src/client/components/GameConfigSettings";
import {
  Difficulty,
  GameMapType,
  GameMode,
  SpecialRuleset,
} from "../../../src/core/game/Game";

describe("GameConfigSettings", () => {
  let settings: GameConfigSettings;

  const baseSettings: GameConfigSettingsData = {
    map: {
      selected: GameMapType.World,
      useRandom: false,
    },
    difficulty: {
      selected: Difficulty.Easy,
      disabled: false,
    },
    gameMode: {
      selected: GameMode.FFA,
    },
    teamCount: {
      selected: 2,
    },
    options: {
      titleKey: "single_modal.options_title",
      bots: {
        value: 10,
        labelKey: "single_modal.bots",
        disabledKey: "single_modal.bots_disabled",
      },
      toggles: [],
      inputCards: [],
    },
    unitTypes: {
      titleKey: "single_modal.enables_title",
      disabledUnits: [],
    },
  };

  beforeEach(async () => {
    settings = document.createElement(
      "game-config-settings",
    ) as GameConfigSettings;
    document.body.appendChild(settings);
    settings.settings = structuredClone(baseSettings);
    await settings.updateComplete;
  });

  afterEach(() => {
    settings.remove();
  });

  it("renders zombie mode cards when the solo flow enables them", async () => {
    settings.settings = {
      ...baseSettings,
      gameMode: {
        selected: GameMode.FFA,
        showZombieOption: true,
      },
    };

    await settings.updateComplete;

    const labels = Array.from(settings.querySelectorAll("button"))
      .map((button) => button.textContent?.trim())
      .filter((label): label is string => Boolean(label));

    expect(labels).toContain("game_mode.zombie_ffa");
    expect(labels).toContain("game_mode.zombie_teams");
  });

  it("emits the zombie ruleset when a zombie card is selected", async () => {
    const eventSpy = vi.fn();
    settings.addEventListener("special-ruleset-selected", eventSpy);
    settings.settings = {
      ...baseSettings,
      gameMode: {
        selected: GameMode.FFA,
        showZombieOption: true,
      },
    };

    await settings.updateComplete;

    const zombieTeamsButton = Array.from(settings.querySelectorAll("button")).find(
      (button) => button.textContent?.includes("game_mode.zombie_teams"),
    );

    expect(zombieTeamsButton).toBeTruthy();
    zombieTeamsButton?.click();

    expect(eventSpy).toHaveBeenCalledTimes(1);
    const event = eventSpy.mock.calls[0][0] as CustomEvent<{
      specialRuleset: SpecialRuleset;
      mode: GameMode;
    }>;
    expect(event.detail).toEqual({
      specialRuleset: SpecialRuleset.ZombieSurvival,
      mode: GameMode.Team,
    });
  });
});
