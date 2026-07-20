import { renderNumber, renderTroops } from "../../src/client/Utils";
import { humanVisibleNumberFloor } from "../../src/headless/OpenFrontGymEnv";

const boundaries = [
  0, 1, 998, 999, 1_000, 1_001, 9_999, 10_000, 10_001, 99_999,
  100_000, 100_001, 999_999, 1_000_000, 1_000_001, 9_999_999,
  10_000_000, 10_000_001,
];

for (const value of boundaries) {
  const quantizedGold = humanVisibleNumberFloor(value);
  if (
    quantizedGold > value ||
    renderNumber(quantizedGold) !== renderNumber(value)
  ) {
    throw new Error(`gold visibility mismatch at ${value}`);
  }

  const troops = value * 10;
  const quantizedTroops = humanVisibleNumberFloor(troops / 10) * 10;
  if (
    quantizedTroops > troops ||
    renderTroops(quantizedTroops) !== renderTroops(troops)
  ) {
    throw new Error(`troop visibility mismatch at ${troops}`);
  }
}

console.log(`human-visible resource boundaries: ${boundaries.length * 2} passed`);
