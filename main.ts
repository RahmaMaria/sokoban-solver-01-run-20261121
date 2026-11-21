// sokoban-solver-01: a push-optimal Sokoban solver, written for this entry.
//
// Idea: the levels are a hallway with independent rooms off it (a box pushed
// out of a room's door can never come back, so rooms never interact). The
// solver therefore splits a level into rooms, solves each room push-optimally
// on its own, and stitches the room solutions together with hallway walks.
// Inside a room it runs A* over pushes: a state is the boxes plus the
// player's reachable region (named by its lowest square), each push costs 1,
// and the heuristic is the cheapest assignment of boxes to goals by
// walking distance. Dead squares (a box there can never reach any goal) and
// frozen 2x2 blocks prune the search, so the first solved state popped is
// push-optimal. If a level does not split into rooms, the same search runs
// over the whole level instead. Everything is budgeted by work (state
// expansions), never by the clock, so two runs write identical bytes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// [dx, dy, walk letter, push letter]. Fixed order keeps every run identical.
const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

// Work budgets (state expansions, never the clock): a room the generator kept
// is solvable far inside its own 250k-state budget, so these caps are slack
// that only guards against a pathological level eating the 60s wall budget.
const ROOM_BUDGET = 500_000;
const WHOLE_BUDGET = 1_500_000;

type Level = {
  w: number;
  h: number;
  wall: Uint8Array; // 1 = wall (outer frame always wall)
  goals: number[]; // sorted
  boxes: number[]; // sorted
  player: number;
};

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  goals.sort((a, b) => a - b);
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

// Squares a lone box can be pushed from onto some goal, found by pulling
// boxes back from the goals: from a live square x, the square y = x - d is
// live when both y and y - d are open floor (the box is pulled onto y while
// the player stands on y - d). A box off a live square is dead forever.
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

// Flood fill from `from` around walls and `occupied` boxes, stamping `seen`
// with `stamp` (a fresh stamp per call avoids clearing the array). Returns
// the lowest square reached: the player's region, named by one number.
function reach(
  level: Level,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  const { w, wall } = level;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (const [dx, dy] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

// True when the box just pushed onto `t` sits in a frozen 2x2 block: four
// squares of walls and boxes, one of them a box off its goal.
function frozenBlock(
  level: Level,
  occupied: Uint8Array,
  goal: Uint8Array,
  t: number,
): boolean {
  const { w, wall } = level;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a])) continue;
    if (!(wall[b] || occupied[b])) continue;
    if (!(wall[c] || occupied[c])) continue;
    if (!(wall[d] || occupied[d])) continue;
    if (
      (occupied[a] && !goal[a]) ||
      (occupied[b] && !goal[b]) ||
      (occupied[c] && !goal[c]) ||
      (occupied[d] && !goal[d])
    ) {
      return true;
    }
  }
  return false;
}

// Shortest walk from `from` to `to` around walls and boxes, lower-case LURD
// letters, first direction first on ties. Throws when unreachable (a bug if
// it ever fires: every push we reconstruct was checked reachable).
function walkTo(
  level: Level,
  occupied: Uint8Array,
  from: number,
  to: number,
): string {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

// The square the player stands on after playing `moves` from `from`.
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) {
    p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  }
  return p;
}

// All index permutations of length n (n <= 4), built once for the assignment
// heuristic.
const PERMS: number[][][] = [[], [[0]]];
for (let n = 2; n <= 4; n++) {
  const out: number[][] = [];
  const build = (used: boolean[], cur: number[]): void => {
    if (cur.length === n) {
      out.push(cur.slice());
      return;
    }
    for (let i = 0; i < n; i++) {
      if (!used[i]) {
        used[i] = true;
        cur.push(i);
        build(used, cur);
        cur.pop();
        used[i] = false;
      }
    }
  };
  build(new Array<boolean>(n).fill(false), []);
  PERMS[n] = out;
}

// A binary heap on [f, -g, order, state]: smallest f first, then deepest,
// then earliest found. Deterministic on every run.
class Heap {
  private a: [number, number, number, number][] = [];
  get size(): number {
    return this.a.length;
  }
  push(e: [number, number, number, number]): void {
    const a = this.a;
    let i = a.length;
    a.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!Heap.less(a[i]!, a[p]!)) break;
      [a[i], a[p]] = [a[p]!, a[i]!];
      i = p;
    }
  }
  pop(): [number, number, number, number] {
    const a = this.a;
    const top = a[0]!;
    const last = a.pop()!;
    if (a.length > 0) {
      a[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= a.length) break;
        if (c + 1 < a.length && Heap.less(a[c + 1]!, a[c]!)) c++;
        if (!Heap.less(a[c]!, a[i]!)) break;
        [a[i], a[c]] = [a[c]!, a[i]!];
        i = c;
      }
    }
    return top;
  }
  private static less(
    x: [number, number, number, number],
    y: [number, number, number, number],
  ): boolean {
    return (
      x[0] < y[0] ||
      (x[0] === y[0] && (x[1] < y[1] || (x[1] === y[1] && x[2] < y[2])))
    );
  }
}

type SearchOpts = {
  // When >= 0, a solved state only counts if the player can still walk to
  // some floor square of this board row afterwards (so it can leave the room
  // it just finished and reach the next one).
  exitRow: number;
};

/**
 * Push-optimal solution of `level` within `maxExpansions` state expansions:
 * {pushes, moves} (moves in LURD, an upper letter marking a push), or null
 * when the budget runs out or the level has no solution. A* over pushes:
 * each push costs 1 and the heuristic is the cheapest assignment of boxes
 * to goals by walking distance (exact for up to 4 boxes, summed nearest
 * goal beyond that), which never overestimates, so the first solved state
 * popped is push-optimal.
 */
function solvePush(
  level: Level,
  maxExpansions: number,
  opts: SearchOpts = { exitRow: -1 },
): { pushes: number; moves: string } | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const n = level.boxes.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(level);
  if (level.boxes.some((b) => !live[b])) return null;

  // Walking distance from every square to each goal, over floor alone.
  const dist = new Int32Array(goals.length * size).fill(-1);
  for (let gi = 0; gi < goals.length; gi++) {
    const base = gi * size;
    const queue = [goals[gi]!];
    dist[base + goals[gi]!] = 0;
    for (let qi = 0; qi < queue.length; qi++) {
      const p = queue[qi]!;
      for (const [dx, dy] of DIRS) {
        const q = p + dy * w + dx;
        if (!wall[q] && dist[base + q] < 0) {
          dist[base + q] = dist[base + p]! + 1;
          queue.push(q);
        }
      }
    }
  }
  // Cheapest assignment of boxes to goals; Infinity when some box can reach
  // no goal at all. Admissible: one push moves one box one square.
  const heuristic = (boxes: number[]): number => {
    if (n === 0) return 0;
    if (n <= 4) {
      let best = Infinity;
      for (const perm of PERMS[n]!) {
        let sum = 0;
        for (let i = 0; i < n; i++) {
          const d = dist[perm[i]! * size + boxes[i]!]!;
          if (d < 0) {
            sum = Infinity;
            break;
          }
          sum += d;
        }
        if (sum < best) best = sum;
      }
      return best;
    }
    let sum = 0;
    for (const b of boxes) {
      let dmin = Infinity;
      for (let gi = 0; gi < goals.length; gi++) {
        const d = dist[gi * size + b]!;
        if (d >= 0 && d < dmin) dmin = d;
      }
      if (dmin === Infinity) return Infinity;
      sum += dmin;
    }
    return sum;
  };

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  // A second stamp array for successor regions: reach() restamps every cell
  // it visits, so computing a successor's region in `seen` would clobber
  // the current state's region stamp the push loop still reads.
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  let markStamp = 0;
  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b]);
  const keyOf = (boxes: number[], low: number): number => {
    let k = low;
    for (const b of boxes) k = k * size + b;
    return k;
  };
  // Can the player, with `boxes` where they are, still walk to the exit row?
  const canExit = (boxes: number[], from: number): boolean => {
    if (opts.exitRow < 0) return true;
    for (const b of boxes) occupied[b] = 1;
    reach(level, occupied, from, seen, ++stamp, stack);
    const mark = stamp;
    for (const b of boxes) occupied[b] = 0;
    for (let x = 0; x < w; x++) {
      const c = opts.exitRow * w + x;
      if (!wall[c] && seen[c] === mark) return true;
    }
    return false;
  };

  const stateBoxes: number[][] = [level.boxes.slice()];
  const statePlayer: number[] = [level.player];
  const stateG: number[] = [0];
  const stateKey: number[] = [];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1];
  const pushTo: number[] = [-1];
  const best = new Map<number, number>();
  for (const b of level.boxes) occupied[b] = 1;
  const startLow = reach(level, occupied, level.player, seen, ++stamp, stack);
  for (const b of level.boxes) occupied[b] = 0;
  const startKey = keyOf(level.boxes, startLow);
  stateKey.push(startKey);
  best.set(startKey, 0);
  const heap = new Heap();
  let order = 0;
  const h0 = heuristic(level.boxes);
  if (h0 === Infinity) return null;
  heap.push([h0, 0, order++, 0]);
  if (solved(level.boxes)) return { pushes: 0, moves: '' };

  let found = -1;
  let expanded = 0;
  while (heap.size > 0 && expanded < maxExpansions) {
    const [, negG, , s] = heap.pop();
    const g = -negG;
    // Stale heap entry: a cheaper route to this state was found since.
    if (best.get(stateKey[s]!) !== s || stateG[s]! !== g) continue;
    const boxes = stateBoxes[s]!;
    if (solved(boxes)) {
      if (canExit(boxes, statePlayer[s]!)) {
        found = s;
        break;
      }
      // Solved but the player cannot leave: keep searching from here, the
      // boxes may be pushed around to a solved state it can leave from.
    }
    expanded++;
    for (const b of boxes) occupied[b] = 1;
    reach(level, occupied, statePlayer[s]!, seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        // The player must stand on b - d to push, and t must be open floor
        // a box can still reach a goal from.
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(level, occupied, goal, t);
        let key = -1;
        let next: number[] | null = null;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((a, bb) => a - bb);
          const low = reach(level, occupied, b, mark, ++markStamp, stack);
          key = keyOf(next, low);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || next === null) continue;
        const prev = best.get(key);
        if (prev !== undefined && stateG[prev]! <= g + 1) continue;
        const hv = heuristic(next);
        if (hv === Infinity) continue;
        const idx = stateBoxes.length;
        stateBoxes.push(next);
        statePlayer.push(b);
        stateG.push(g + 1);
        stateKey.push(key);
        parent.push(s);
        pushFrom.push(b);
        pushTo.push(t);
        best.set(key, idx);
        heap.push([g + 1 + hv, -(g + 1), order++, idx]);
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;

  // Reconstruct the moves: for each push on the path, walk to the push
  // square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes.slice();
  for (const s of path) {
    const nextBoxes = stateBoxes[s]!;
    const from = pushFrom[s]!;
    const to = pushTo[s]!;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d)!;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(level, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += dir[3];
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

type Room = { squares: number[]; boxes: number[]; goals: number[] };

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway:
 * a box there reaches no goal) out of the floor, and each remaining part
 * holding a box is a room. Null when a box starts in the hallway, a goal is
 * cut off, or some part holds a different count of boxes and goals: then
 * the level is no set of rooms.
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) {
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) {
    if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  }
  if (goals.some((g) => part[g]! < 0)) return null;
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/**
 * The level solved room by room: each room's push-optimal solution in turn,
 * the other rooms' boxes standing as walls (on their goals once their room
 * is done), the player walking on from where the last room left it. A box
 * never leaves its room, so the rooms' optima add up to the level's.
 * Null when a room's search gives up; the caller then tries the whole level.
 */
function solveByRooms(level: Level): { pushes: number; moves: string } | null {
  const rooms = roomsOf(level);
  if (rooms === null) return null;
  const exitRow = Math.floor(level.player / level.w);
  let player = level.player;
  let pushes = 0;
  let moves = '';
  const done = new Set<Room>();
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of done.has(other) ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved = solvePush(sub, ROOM_BUDGET, { exitRow });
    if (solved === null) return null;
    pushes += solved.pushes;
    moves += solved.moves;
    player = endOf(level.w, player, solved.moves);
    done.add(room);
  }
  return { pushes, moves };
}

function solveLevel(text: string): string {
  const level = parseLevel(text);
  const byRooms = solveByRooms(level);
  if (byRooms !== null) return byRooms.moves;
  const whole = solvePush(level, WHOLE_BUDGET);
  if (whole !== null) return whole.moves;
  return '';
}

for (const id of readdirSync(INPUT).sort()) {
  let moves = '';
  try {
    moves = solveLevel(readFileSync(join(INPUT, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(OUTPUT, id), { recursive: true });
  writeFileSync(join(OUTPUT, id, 'solution.txt'), `${moves}\n`);
}
