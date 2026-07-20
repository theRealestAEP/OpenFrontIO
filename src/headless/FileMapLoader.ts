import { readFile } from "node:fs/promises";
import path from "node:path";
import { GameMapType } from "../core/game/Game";
import { GameMapLoader, MapData } from "../core/game/GameMapLoader";
import { MapManifest } from "../core/game/TerrainMapLoader";

export class FileMapLoader implements GameMapLoader {
  private maps = new Map<GameMapType, MapData>();

  constructor(private readonly mapsRoot: string) {}

  getMapData(map: GameMapType): MapData {
    const cached = this.maps.get(map);
    if (cached !== undefined) return cached;

    const key = Object.keys(GameMapType).find(
      (k) => GameMapType[k as keyof typeof GameMapType] === map,
    );
    const mapDir = key?.toLowerCase();
    if (mapDir === undefined) {
      throw new Error(`Unknown map: ${map}`);
    }

    const filePath = (file: string) => path.join(this.mapsRoot, mapDir, file);
    const data = {
      mapBin: () => this.readBinary(filePath("map.bin")),
      map4xBin: () => this.readBinary(filePath("map4x.bin")),
      map16xBin: () => this.readBinary(filePath("map16x.bin")),
      manifest: () => this.readJson<MapManifest>(filePath("manifest.json")),
      webpPath: filePath("thumbnail.webp"),
    } satisfies MapData;
    this.maps.set(map, data);
    return data;
  }

  private async readBinary(file: string): Promise<Uint8Array> {
    return new Uint8Array(await readFile(file));
  }

  private async readJson<T>(file: string): Promise<T> {
    return JSON.parse(await readFile(file, "utf8")) as T;
  }
}
