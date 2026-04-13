import {
  CureState,
  Execution,
  Game,
  GameMode,
  Player,
  Team,
  UnitType,
} from "../game/Game";
import {
  isZombiePlayer,
  ZOMBIE_CURE_RESEARCH_TICKS,
  zombieResearchGroupKey,
} from "../game/ZombieUtils";

export class ZombieSurvivalExecution implements Execution {
  private active = true;
  private mg!: Game;
  private cureTicksRemaining = new Map<string, number>();
  private curedGroups = new Set<string>();

  init(mg: Game): void {
    this.mg = mg;
  }

  tick(ticks: number): void {
    if (this.mg.getWinner() !== null) {
      this.active = false;
      return;
    }

    this.updateCureProgress();

    const livingZombies = this.mg
      .players()
      .filter((player) => isZombiePlayer(player));
    if (livingZombies.length === 0) {
      this.declareWinner();
      this.active = false;
    }
  }

  private updateCureProgress(): void {
    const nonZombiePlayers = this.mg
      .allPlayers()
      .filter((player) => !isZombiePlayer(player));
    const groups = new Map<string, Player[]>();

    for (const player of nonZombiePlayers) {
      const key = zombieResearchGroupKey(player);
      const players = groups.get(key);
      if (players) {
        players.push(player);
      } else {
        groups.set(key, [player]);
      }
    }

    for (const [groupKey, players] of groups.entries()) {
      if (this.curedGroups.has(groupKey)) {
        for (const player of players) {
          player.setCureState(CureState.Cured);
          player.setCureProgressRemainingTicks(null);
        }
        continue;
      }

      const readyLab = players
        .flatMap((player) => player.units(UnitType.ResearchLab))
        .find(
          (unit) =>
            unit.isActive() && !unit.isRuined() && !unit.isUnderConstruction(),
        );

      if (!readyLab) {
        this.cureTicksRemaining.delete(groupKey);
        for (const player of players) {
          player.setCureState(CureState.None);
          player.setCureProgressRemainingTicks(null);
        }
        continue;
      }

      const remaining =
        (this.cureTicksRemaining.get(groupKey) ?? ZOMBIE_CURE_RESEARCH_TICKS) -
        1;
      if (remaining <= 0) {
        this.cureTicksRemaining.delete(groupKey);
        this.curedGroups.add(groupKey);
        for (const player of players) {
          player.setCureState(CureState.Cured);
          player.setCureProgressRemainingTicks(null);
        }
        continue;
      }

      this.cureTicksRemaining.set(groupKey, remaining);
      for (const player of players) {
        player.setCureState(CureState.Researching);
        player.setCureProgressRemainingTicks(remaining);
      }
    }
  }

  private declareWinner(): void {
    const survivors = this.mg
      .players()
      .filter((player) => !isZombiePlayer(player));
    if (survivors.length === 0) {
      return;
    }

    if (this.mg.config().gameConfig().gameMode === GameMode.FFA) {
      const winner = survivors.sort(
        (a, b) => b.numTilesOwned() - a.numTilesOwned(),
      )[0];
      if (winner) {
        this.mg.setWinner(winner, this.mg.stats().stats());
      }
      return;
    }

    const teamToTiles = new Map<Team, number>();
    for (const player of survivors) {
      const team = player.team();
      if (team === null) {
        continue;
      }
      teamToTiles.set(
        team,
        (teamToTiles.get(team) ?? 0) + player.numTilesOwned(),
      );
    }

    const winner = Array.from(teamToTiles.entries()).sort(
      (a, b) => b[1] - a[1],
    )[0];
    if (winner) {
      this.mg.setWinner(winner[0], this.mg.stats().stats());
    }
  }

  isActive(): boolean {
    return this.active;
  }

  activeDuringSpawnPhase(): boolean {
    return false;
  }
}
