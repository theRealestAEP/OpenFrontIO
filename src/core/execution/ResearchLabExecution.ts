import { Execution, Game, Unit } from "../game/Game";

export class ResearchLabExecution implements Execution {
  private active = true;

  constructor(private readonly lab: Unit) {}

  init(mg: Game, ticks: number): void {}

  tick(ticks: number): void {
    if (!this.lab.isActive() || this.lab.isRuined()) {
      this.active = false;
    }
  }

  isActive(): boolean {
    return this.active;
  }

  activeDuringSpawnPhase(): boolean {
    return false;
  }
}
