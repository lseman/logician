declare module "bun:sqlite" {
  class Database {
    constructor(filename: string);
    close(): void;
    exec(sql: string): void;
    prepare(sql: string): Statement;
    query(sql: string): IterableIterator<any>;
    run(sql: string, ...params: any[]): { changes: number; lastInsertRowid: bigint | number };
    get(sql: string, ...params: any[]): any;
    all(sql: string, ...params: any[]): any[];
  }
  
  interface Statement {
    run(...params: any[]): { changes: number; lastInsertRowid: bigint | number };
    get(...params: any[]): any;
    all(...params: any[]): any[];
    iterate(): IterableIterator<any>;
  }

  export { Database, Statement };
}
