import pygame
import sys
import math
import random
from collections import deque

# ======================== CONFIGURATION ========================
WIDTH       = 900
HEIGHT      = 700
UI_HEIGHT   = 60
TILE_SIZE   = 40
GRID_COLS   = WIDTH // TILE_SIZE                         # 22
GRID_ROWS   = (HEIGHT - UI_HEIGHT) // TILE_SIZE          # 16
FPS         = 60
Y_OFFSET    = UI_HEIGHT

# ======================== COLORS ========================
BLACK         = (5, 5, 15)
WHITE         = (255, 255, 255)
GRAY          = (100, 100, 120)
WALL_FILL     = (50, 50, 70)
WALL_EDGE     = (80, 80, 110)
WALL_TOP      = (105, 105, 140)
FLOOR_COLOR   = (20, 20, 32)
FLOOR_GRID    = (28, 28, 42)
UI_BG         = (12, 12, 22)
ACCENT        = (180, 50, 255)
EXIT_COLOR    = (50, 255, 100)
EXIT_LOCKED   = (40, 80, 50)

PORTAL_DEFS = [
    ((255,  70,  70), "Red"),
    (( 70, 150, 255), "Blue"),
    (( 70, 255, 130), "Green"),
    ((255, 215,  60), "Yellow"),
    ((200,  80, 255), "Purple"),
    ((255, 140,  50), "Orange"),
    (( 60, 230, 230), "Cyan"),
]

# ======================== GAME STATES ========================
START_SCREEN     = 0
PLAYING          = 1
PAUSED           = 2
LEVEL_COMPLETE   = 3
GAME_COMPLETE    = 4


# ======================== MCTS VALIDATOR ========================
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_actions = None  # Lazy initialization

class MCTSValidator:
    """
    Monte Carlo Tree Search to validate that a generated level is solvable.
    It explores moves: (Up, Down, Left, Right).
    State: (x, y, current_step_in_sequence)
    If it touches a wrong portal, the branch is a dead end.
    If it reaches the exit after all portals, the branch wins.
    """
    def __init__(self, grid, portals, start, exit_pos, sequence):
        self.grid = grid
        self.portals = portals
        self.start = start
        self.exit = exit_pos
        self.sequence = sequence
        self.max_depth = 150 # Max moves per rollout

    def get_valid_actions(self, state):
        if state[2] == -1: 
            return [] # Dead end state
        x, y, step = state
        actions = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_COLS and 0 <= ny < GRID_ROWS:
                if self.grid[ny][nx] != '#':
                    actions.append((dx, dy))
        return actions

    def apply_action(self, state, action):
        x, y, step = state
        nx, ny = x + action[0], y + action[1]
        cell = self.grid[ny][nx]
        
        if cell == 'E':
            if step == len(self.sequence):
                return (nx, ny, step), True # WIN!
            return (nx, ny, step), False
        elif cell == 'P':
            p = self.portals[(nx, ny)]
            if p['index'] == self.sequence[step]:
                dest_x, dest_y = p['dest']
                return (dest_x, dest_y, step + 1), False
            else:
                return (nx, ny, -1), False # Dead end (touched wrong portal)
        return (nx, ny, step), False

    def simulate(self, state):
        if state[2] == -1: 
            return 0
        current = state
        for _ in range(self.max_depth):
            actions = self.get_valid_actions(current)
            if not actions: 
                return 0
            action = random.choice(actions)
            current, won = self.apply_action(current, action)
            if current[2] == -1: 
                return 0
            if won: 
                return 1
        return 0

    def validate(self, max_iterations=2000):
        root = MCTSNode(self.start + (0,))
        
        for _ in range(max_iterations):
            node = root
            state = node.state
            
            # Selection
            while node.untried_actions is not None and not node.untried_actions and node.children:
                c = 1.414 # Exploration constant
                best_child = max(node.children, key=lambda child: child.wins / max(1, child.visits) + c * math.sqrt(math.log(max(1, node.visits)) / max(1, child.visits)))
                node = best_child
                state = node.state
            
            # Expansion
            if node.untried_actions is None:
                node.untried_actions = self.get_valid_actions(state)
            
            if node.untried_actions:
                action = node.untried_actions.pop()
                new_state, won = self.apply_action(state, action)
                if won: 
                    return True # Found a guaranteed win during expansion
                child = MCTSNode(new_state, node)
                node.children.append(child)
                node = child
                state = new_state
            
            # Simulation (Rollout)
            win = self.simulate(state)
            if win: 
                return True # Found a winning random path!
            
            # Backpropagation
            while node is not None:
                node.visits += 1
                node.wins += win
                node = node.parent
                
        return False # No solution found in budget


# ======================== SOUND ENGINE ========================
class SoundEngine:
    def __init__(self):
        self.enabled = False
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.enabled = True
        except Exception:
            return
        self.sfx = {
            "teleport":  self._sweep(200, 900, 0.35, 0.35),
            "wrong":     self._sweep(400, 80, 0.45, 0.35, wave="saw"),
            "win":       self._chord([523, 659, 784], 0.6, 0.25),
            "level_up":  self._sweep(400, 1200, 0.5, 0.3),
            "exit":      self._chord([523, 659, 784, 1047], 0.8, 0.3),
            "click":     self._tone(800, 0.05, 0.15),
            "pause":     self._tone(300, 0.1, 0.15),
        }

    def _tone(self, freq, dur, vol=0.3, wave="sine"):
        if not self.enabled: return None
        sr = 22050; n = int(sr * dur); buf = bytearray(n * 2)
        for i in range(n):
            t = i / sr
            if wave == "sine": s = math.sin(2 * math.pi * freq * t)
            elif wave == "square": s = 1.0 if math.sin(2 * math.pi * freq * t) > 0 else -1.0
            elif wave == "saw": s = 2 * (t * freq - math.floor(t * freq + 0.5))
            else: s = math.sin(2 * math.pi * freq * t)
            fade = 1.0 - (i / n) ** 1.5
            val = int(vol * 32767 * s * fade)
            buf[i*2] = val & 0xFF; buf[i*2 + 1] = (val >> 8) & 0xFF
        return pygame.mixer.Sound(buffer=bytes(buf))

    def _sweep(self, f1, f2, dur, vol=0.3, wave="sine"):
        if not self.enabled: return None
        sr = 22050; n = int(sr * dur); buf = bytearray(n * 2)
        for i in range(n):
            t = i / n; freq = f1 + (f2 - f1) * t
            s = math.sin(2 * math.pi * freq * (i / sr))
            if wave == "saw": s = 2 * (freq * (i / sr) - math.floor(freq * (i / sr) + 0.5))
            fade = 1.0 - t ** 1.5
            val = int(vol * 32767 * s * fade)
            buf[i*2] = val & 0xFF; buf[i*2 + 1] = (val >> 8) & 0xFF
        return pygame.mixer.Sound(buffer=bytes(buf))

    def _chord(self, freqs, dur, vol=0.25):
        if not self.enabled: return None
        sr = 22050; n = int(sr * dur); buf = bytearray(n * 2)
        for i in range(n):
            t = i / n
            s = sum(math.sin(2 * math.pi * f * (i / sr)) for f in freqs) / len(freqs)
            fade = 1.0 - t ** 1.5
            val = int(vol * 32767 * s * fade)
            buf[i*2] = val & 0xFF; buf[i*2 + 1] = (val >> 8) & 0xFF
        return pygame.mixer.Sound(buffer=bytes(buf))

    def play(self, name):
        if self.enabled and name in self.sfx and self.sfx[name]:
            self.sfx[name].play()

    def toggle(self):
        if self.enabled:
            pygame.mixer.quit(); self.enabled = False
        else:
            try: pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512); self.enabled = True
            except Exception: pass


# ======================== PARTICLE SYSTEM ========================
class Particle:
    __slots__ = ("x", "y", "vx", "vy", "color", "life", "max_life", "size")
    def __init__(self, x, y, vx, vy, color, life, size=3):
        self.x = x; self.y = y; self.vx = vx; self.vy = vy
        self.color = color; self.life = life; self.max_life = life; self.size = size

    def update(self, dt):
        self.x += self.vx * dt; self.y += self.vy * dt
        self.vx *= 0.94; self.vy *= 0.94; self.life -= dt

    def draw(self, surf):
        if self.life <= 0: return
        a = self.life / self.max_life
        r = max(1, int(self.size * a))
        c = (int(self.color[0]*a), int(self.color[1]*a), int(self.color[2]*a))
        pygame.draw.circle(surf, c, (int(self.x), int(self.y)), r)

class ParticleSystem:
    def __init__(self): self.particles = []
    def burst(self, x, y, color, count=24, speed=250, life=0.6, size=5):
        for _ in range(count):
            ang = random.uniform(0, 2 * math.pi)
            spd = random.uniform(speed * 0.3, speed)
            self.particles.append(Particle(x, y, math.cos(ang) * spd, math.sin(ang) * spd, color, life, random.randint(2, size)))
    def ring(self, x, y, color, count=30, speed=180, life=0.7, size=3):
        for i in range(count):
            ang = (i / count) * 2 * math.pi
            self.particles.append(Particle(x, y, math.cos(ang) * speed, math.sin(ang) * speed, color, life, size))
    def update(self, dt):
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles: p.update(dt)
    def draw(self, surf):
        for p in self.particles: p.draw(surf)
    def clear(self): self.particles.clear()


# ======================== PLAYER ========================
class Player:
    def __init__(self, x, y):
        self.pos = pygame.math.Vector2(x, y); self.vel = pygame.math.Vector2(0, 0)
        self.speed = 220; self.radius = 14; self.trail = []; self.max_trail = 14
        self.rect = pygame.Rect(0, 0, self.radius * 2, self.radius * 2)
        self.rect.center = (int(x), int(y))
        self.teleport_cooldown = 0.0

    def update(self, dt, walls):
        if self.teleport_cooldown > 0: self.teleport_cooldown -= dt
        keys = pygame.key.get_pressed()
        dx = (keys[pygame.K_RIGHT] or keys[pygame.K_d]) - (keys[pygame.K_LEFT] or keys[pygame.K_a])
        dy = (keys[pygame.K_DOWN]  or keys[pygame.K_s]) - (keys[pygame.K_UP]   or keys[pygame.K_w])
        self.vel.x = dx; self.vel.y = dy
        if self.vel.length() > 0: self.vel = self.vel.normalize() * self.speed

        self.pos.x += self.vel.x * dt; self.rect.centerx = int(self.pos.x)
        for w in walls:
            if self.rect.colliderect(w):
                if self.vel.x > 0: self.rect.right = w.left
                elif self.vel.x < 0: self.rect.left = w.right
                self.pos.x = self.rect.centerx

        self.pos.y += self.vel.y * dt; self.rect.centery = int(self.pos.y)
        for w in walls:
            if self.rect.colliderect(w):
                if self.vel.y > 0: self.rect.bottom = w.top
                elif self.vel.y < 0: self.rect.top = w.bottom
                self.pos.y = self.rect.centery

        self.trail.append((self.pos.x, self.pos.y))
        if len(self.trail) > self.max_trail: self.trail.pop(0)

    def draw(self, surf):
        for i, (tx, ty) in enumerate(self.trail):
            a = (i + 1) / len(self.trail)
            r = max(1, int(self.radius * a * 0.55))
            c = (int(ACCENT[0]*a*0.4), int(ACCENT[1]*a*0.4), int(ACCENT[2]*a*0.4))
            pygame.draw.circle(surf, c, (int(tx), int(ty)), r)
        cx, cy = int(self.pos.x), int(self.pos.y)
        pygame.draw.circle(surf, WHITE, (cx, cy), self.radius)
        pygame.draw.circle(surf, ACCENT, (cx, cy), self.radius, 2)
        pygame.draw.circle(surf, (200, 150, 255), (cx, cy), self.radius // 2)


# ======================== PORTAL & EXIT ========================
class Portal:
    def __init__(self, x, y, color, name, index):
        self.pos = pygame.math.Vector2(x, y); self.color = color; self.name = name; self.index = index
        self.radius = 22; self.rect = pygame.Rect(0, 0, self.radius * 2, self.radius * 2)
        self.rect.center = (int(x), int(y))
        self.active = False; self.completed = False; self.angle = 0.0
        self.orbit_particles = []; self.dest_x = x; self.dest_y = y

    def update(self, dt):
        self.angle += dt * 3.5
        if self.active and random.random() < 0.6:
            ang = random.uniform(0, 2 * math.pi); r = self.radius + random.uniform(-4, 6)
            self.orbit_particles.append({"x": self.pos.x + math.cos(ang) * r, "y": self.pos.y + math.sin(ang) * r, "vx": math.cos(ang) * 60, "vy": math.sin(ang) * 60, "life": 0.5, "max_life": 0.5})
        for p in self.orbit_particles:
            p["x"] += p["vx"] * dt; p["y"] += p["vy"] * dt; p["life"] -= dt
        self.orbit_particles = [p for p in self.orbit_particles if p["life"] > 0]

    def draw(self, surf, time_ms):
        cx, cy = int(self.pos.x), int(self.pos.y)
        if self.completed:
            dim = tuple(c // 5 for c in self.color)
            pygame.draw.circle(surf, dim, (cx, cy), self.radius, 2)
            pygame.draw.circle(surf, (60, 120, 60), (cx, cy), 6); return
        if self.active:
            pulse = math.sin(time_ms * 0.006) * 4
            for r in range(self.radius + 18, self.radius, -3):
                a = (self.radius + 18 - r) / 18
                c = (int(self.color[0]*a*0.35), int(self.color[1]*a*0.35), int(self.color[2]*a*0.35))
                pygame.draw.circle(surf, c, (cx, cy), r)
            pygame.draw.circle(surf, self.color, (cx, cy), int(self.radius + pulse))
            pygame.draw.circle(surf, WHITE, (cx, cy), self.radius - 4, 2)
            for i in range(3):
                a = self.angle + i * (2 * math.pi / 3)
                x1 = cx + math.cos(a) * (self.radius - 6); y1 = cy + math.sin(a) * (self.radius - 6)
                x2 = cx + math.cos(a + 0.9) * (self.radius - 6); y2 = cy + math.sin(a + 0.9) * (self.radius - 6)
                pygame.draw.line(surf, WHITE, (x1, y1), (x2, y2), 2)
        else:
            dim = tuple(c // 4 for c in self.color)
            pygame.draw.circle(surf, dim, (cx, cy), self.radius, 2)
            pygame.draw.circle(surf, dim, (cx, cy), 5)
        for p in self.orbit_particles:
            a = p["life"] / p["max_life"]
            c = (int(self.color[0]*a), int(self.color[1]*a), int(self.color[2]*a))
            pygame.draw.circle(surf, c, (int(p["x"]), int(p["y"])), 2)

class ExitGate:
    def __init__(self, x, y):
        self.pos = pygame.math.Vector2(x, y); self.radius = 26
        self.rect = pygame.Rect(0, 0, self.radius * 2, self.radius * 2); self.rect.center = (int(x), int(y))
        self.active = False; self.angle = 0.0

    def update(self, dt): self.angle += dt * 2.0

    def draw(self, surf, time_ms):
        cx, cy = int(self.pos.x), int(self.pos.y)
        if self.active:
            pulse = math.sin(time_ms * 0.005) * 5
            for r in range(self.radius + 20, self.radius, -3):
                a = (self.radius + 20 - r) / 20
                c = (int(EXIT_COLOR[0]*a*0.35), int(EXIT_COLOR[1]*a*0.35), int(EXIT_COLOR[2]*a*0.35))
                pygame.draw.circle(surf, c, (cx, cy), r)
            pygame.draw.circle(surf, EXIT_COLOR, (cx, cy), int(self.radius + pulse))
            pygame.draw.circle(surf, WHITE, (cx, cy), self.radius - 4, 2)
            for i in range(5):
                a = self.angle + i * (2 * math.pi / 5)
                x1 = cx + math.cos(a) * (self.radius - 8); y1 = cy + math.sin(a) * (self.radius - 8)
                pygame.draw.circle(surf, WHITE, (int(x1), int(y1)), 3)
        else:
            pygame.draw.circle(surf, EXIT_LOCKED, (cx, cy), self.radius, 2)
            pygame.draw.rect(surf, EXIT_LOCKED, (cx - 8, cy - 2, 16, 12), 2)
            pygame.draw.arc(surf, EXIT_LOCKED, (cx - 6, cy - 10, 12, 12), 0, math.pi, 2)


# ======================== LEVEL GENERATOR ========================
class LevelGenerator:
    def __init__(self, difficulty):
        self.difficulty = difficulty
        self.num_portals = min(3 + difficulty, len(PORTAL_DEFS))
        self.num_rooms = self.num_portals + random.randint(2, 4)
        self.room_min = max(3, 6 - difficulty // 2)
        self.room_max = max(5, 9 - difficulty // 2)
        self.corridor_wide = difficulty <= 2
        self.extra_loops = max(0, difficulty - 1)

    def generate(self):
        for _attempt in range(30):
            grid, rooms = self._place_rooms()
            if len(rooms) < max(3, self.num_portals): continue
            self._connect_rooms(grid, rooms)
            result = self._populate(grid, rooms)
            if result is not None: return result
        return self._fallback()

    def _place_rooms(self):
        grid = [[1] * GRID_COLS for _ in range(GRID_ROWS)]; rooms = []; tries = 0
        while len(rooms) < self.num_rooms and tries < 300:
            tries += 1
            w = random.randint(self.room_min, self.room_max); h = random.randint(self.room_min, self.room_max)
            x = random.randint(1, GRID_COLS - w - 2); y = random.randint(1, GRID_ROWS - h - 2)
            room = pygame.Rect(x, y, w, h)
            if any(room.inflate(2, 2).colliderect(r) for r in rooms): continue
            rooms.append(room)
            for ry in range(room.top, room.bottom):
                for rx in range(room.left, room.right): grid[ry][rx] = 0
        return grid, rooms

    def _connect_rooms(self, grid, rooms):
        for i in range(len(rooms) - 1): self._carve_corridor(grid, rooms[i], rooms[i + 1])
        for _ in range(self.extra_loops):
            if len(rooms) >= 2:
                a, b = random.sample(range(len(rooms)), 2)
                self._carve_corridor(grid, rooms[a], rooms[b])

    def _carve_corridor(self, grid, r1, r2):
        x1, y1 = r1.centerx, r1.centery; x2, y2 = r2.centerx, r2.centery
        if random.random() < 0.5: self._carve_h(grid, x1, x2, y1); self._carve_v(grid, y1, y2, x2)
        else: self._carve_v(grid, y1, y2, x1); self._carve_h(grid, x1, x2, y2)

    def _carve_h(self, grid, x1, x2, y):
        lo, hi = min(x1, x2), max(x1, x2)
        for x in range(lo, hi + 1):
            self._open(grid, x, y)
            if self.corridor_wide: self._open(grid, x, y + 1)

    def _carve_v(self, grid, y1, y2, x):
        lo, hi = min(y1, y2), max(y1, y2)
        for y in range(lo, hi + 1):
            self._open(grid, x, y)
            if self.corridor_wide: self._open(grid, x + 1, y)

    @staticmethod
    def _open(grid, x, y):
        if 0 <= x < GRID_COLS and 0 <= y < GRID_ROWS: grid[y][x] = 0

    def _populate(self, grid, rooms):
        if len(rooms) < 3: return None
        rooms.sort(key=lambda r: r.topleft)
        start_room = rooms[0]; exit_room = rooms[-1]
        candidate_rooms = [r for r in rooms if r is not start_room and r is not exit_room]
        if len(candidate_rooms) < self.num_portals: candidate_rooms = rooms[1:]
        if len(candidate_rooms) < self.num_portals: return None

        portal_rooms = random.sample(candidate_rooms, self.num_portals)
        portals = []
        for i in range(self.num_portals):
            room = portal_rooms[i]; color, name = PORTAL_DEFS[i]
            px = random.randint(room.left + 1, room.right - 2); py = random.randint(room.top + 1, room.bottom - 2)
            wx = px * TILE_SIZE + TILE_SIZE // 2; wy = py * TILE_SIZE + TILE_SIZE // 2 + Y_OFFSET
            portals.append(Portal(wx, wy, color, name, i))

        all_rooms = rooms[:]
        for p in portals:
            valid = [r for r in all_rooms if not r.collidepoint((p.pos.x - TILE_SIZE//2) / TILE_SIZE, (p.pos.y - Y_OFFSET - TILE_SIZE//2) / TILE_SIZE)]
            if not valid: valid = all_rooms
            dest = random.choice(valid)
            p.dest_x = dest.centerx * TILE_SIZE + TILE_SIZE // 2; p.dest_y = dest.centery * TILE_SIZE + TILE_SIZE // 2 + Y_OFFSET

        sequence = list(range(self.num_portals)); random.shuffle(sequence)

        # --- MCTS VALIDATION ---
        mcts_grid = [['#' if grid[y][x] == 1 else '.' for x in range(GRID_COLS)] for y in range(GRID_ROWS)]
        portal_dict = {}
        for p in portals:
            gx = int((p.pos.x - TILE_SIZE//2) / TILE_SIZE); gy = int((p.pos.y - Y_OFFSET - TILE_SIZE//2) / TILE_SIZE)
            mcts_grid[gy][gx] = 'P'
            dest_gx = int((p.dest_x - TILE_SIZE//2) / TILE_SIZE); dest_gy = int((p.dest_y - Y_OFFSET - TILE_SIZE//2) / TILE_SIZE)
            portal_dict[(gx, gy)] = {'dest': (dest_gx, dest_gy), 'index': p.index}
        
        sx, sy = start_room.centerx, start_room.centery
        ex, ey = exit_room.centerx, exit_room.centery
        mcts_grid[sy][sx] = '.'; mcts_grid[ey][ex] = 'E'
        
        validator = MCTSValidator(mcts_grid, portal_dict, (sx, sy), (ex, ey), sequence)
        if not validator.validate(max_iterations=2000):
            return None # Generation failed MCTS, retry

        sx_px = sx * TILE_SIZE + TILE_SIZE // 2; sy_px = sy * TILE_SIZE + TILE_SIZE // 2 + Y_OFFSET
        ex_px = ex * TILE_SIZE + TILE_SIZE // 2; ey_px = ey * TILE_SIZE + TILE_SIZE // 2 + Y_OFFSET

        walls = self._grid_to_walls(grid)
        return walls, portals, (sx_px, sy_px), (ex_px, ey_px), sequence

    @staticmethod
    def _grid_to_walls(grid):
        walls = []
        for y in range(GRID_ROWS):
            for x in range(GRID_COLS):
                if grid[y][x] == 1:
                    walls.append(pygame.Rect(x * TILE_SIZE, y * TILE_SIZE + Y_OFFSET, TILE_SIZE, TILE_SIZE))
        return walls

    def _fallback(self):
        grid = [[1] * GRID_COLS for _ in range(GRID_ROWS)]
        for y in range(1, GRID_ROWS - 1):
            for x in range(1, GRID_COLS - 1): grid[y][x] = 0
        walls = self._grid_to_walls(grid)
        portals = []
        for i in range(min(3, self.num_portals)):
            color, name = PORTAL_DEFS[i]
            wx = (3 + i * 5) * TILE_SIZE + TILE_SIZE // 2; wy = 5 * TILE_SIZE + TILE_SIZE // 2 + Y_OFFSET
            p = Portal(wx, wy, color, name, i)
            p.dest_x = (10 + i * 2) * TILE_SIZE + TILE_SIZE // 2; p.dest_y = 8 * TILE_SIZE + TILE_SIZE // 2 + Y_OFFSET
            portals.append(p)
        sequence = list(range(len(portals))); random.shuffle(sequence)
        sx = 2 * TILE_SIZE + TILE_SIZE // 2; sy = 2 * TILE_SIZE + TILE_SIZE // 2 + Y_OFFSET
        ex = (GRID_COLS - 3) * TILE_SIZE + TILE_SIZE // 2; ey = (GRID_ROWS - 3) * TILE_SIZE + TILE_SIZE // 2 + Y_OFFSET
        return walls, portals, (sx, sy), (ex, ey), sequence


# ======================== GAME ========================
class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Portal Sequence Puzzle - MCTS Validated")
        self.clock = pygame.time.Clock()

        self.font_lg = pygame.font.SysFont("Consolas", 48, bold=True)
        self.font_md = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_sm = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_xs = pygame.font.SysFont("Consolas", 13)

        self.sound = SoundEngine()
        self.particles = ParticleSystem()

        self.state = START_SCREEN
        self.level = 1; self.difficulty = 1
        self.level_time = 0.0; self.total_time = 0.0
        self.hint_active = False; self.hint_timer = 0.0
        self.shake_intensity = 0; self.shake_timer = 0.0
        self.flash_color = None; self.flash_timer = 0.0

        self.walls = []; self.portals = []; self.player = None; self.exit_gate = None
        self.start_pos = (0, 0); self.sequence = []; self.current_step = 0
        self._generate_level()

    def _generate_level(self):
        gen = LevelGenerator(self.difficulty)
        self.walls, self.portals, self.start_pos, exit_pos, self.sequence = gen.generate()
        self.player = Player(self.start_pos[0], self.start_pos[1])
        self.exit_gate = ExitGate(exit_pos[0], exit_pos[1])
        self.current_step = 0; self.particles.clear()
        self.hint_active = False; self.hint_timer = 0
        if self.portals: self.portals[self.sequence[0]].active = True

    def _reset_puzzle(self):
        self.player.pos = pygame.math.Vector2(self.start_pos[0], self.start_pos[1])
        self.player.trail.clear(); self.current_step = 0
        for p in self.portals: p.active = False; p.completed = False
        if self.portals: self.portals[self.sequence[0]].active = True
        self.exit_gate.active = False

    def _shake(self, intensity, duration=0.4):
        self.shake_intensity = max(self.shake_intensity, intensity)
        self.shake_timer = max(self.shake_timer, duration)

    def _flash(self, color, duration=0.3):
        self.flash_color = color; self.flash_timer = duration

    def _handle_event(self, event):
        if event.type == pygame.QUIT: return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if self.state == PLAYING: self.state = PAUSED; self.sound.play("pause")
                elif self.state == PAUSED: self.state = PLAYING; self.sound.play("pause")
                elif self.state == START_SCREEN: return False
            elif event.key == pygame.K_m: self.sound.toggle()
            elif event.key == pygame.K_SPACE:
                if self.state == START_SCREEN: self.state = PLAYING; self.sound.play("click")
                elif self.state == LEVEL_COMPLETE:
                    self.level += 1; self.difficulty = min(5, 1 + (self.level - 1))
                    self._generate_level(); self.state = PLAYING; self.sound.play("click")
                elif self.state == GAME_COMPLETE:
                    self.level = 1; self.difficulty = 1; self.total_time = 0
                    self._generate_level(); self.state = PLAYING; self.sound.play("click")
            elif event.key == pygame.K_r:
                if self.state in (PLAYING, PAUSED):
                    self._generate_level(); self.state = PLAYING; self.sound.play("click")
                elif self.state == GAME_COMPLETE:
                    self.level = 1; self.difficulty = 1; self.total_time = 0
                    self._generate_level(); self.state = PLAYING
            elif event.key == pygame.K_p and self.state == PLAYING: self.state = PAUSED; self.sound.play("pause")
            elif event.key == pygame.K_p and self.state == PAUSED: self.state = PLAYING; self.sound.play("pause")
            elif event.key == pygame.K_h and self.state == PLAYING:
                if not self.hint_active and self.current_step < len(self.sequence):
                    self.hint_active = True; self.hint_timer = 2.0
                    self.level_time += 5.0; self.sound.play("click")
        return True

    def _update(self, dt):
        if self.state != PLAYING: return
        self.level_time += dt; self.total_time += dt
        if self.hint_timer > 0:
            self.hint_timer -= dt
            if self.hint_timer <= 0: self.hint_active = False
        if self.shake_timer > 0: self.shake_timer -= dt
        else: self.shake_intensity = 0
        if self.flash_timer > 0: self.flash_timer -= dt

        self.player.update(dt, self.walls)
        self.particles.update(dt)
        for p in self.portals: p.update(dt)
        self.exit_gate.update(dt)

        if self.player.teleport_cooldown <= 0:
            for portal in self.portals:
                if self.player.rect.colliderect(portal.rect):
                    self._on_portal_touch(portal); break

        if self.exit_gate.active and self.player.rect.colliderect(self.exit_gate.rect):
            self._on_exit_reached()

    def _on_portal_touch(self, portal):
        if portal.completed: return
        expected_index = self.sequence[self.current_step]
        if portal.index == expected_index:
            portal.completed = True; portal.active = False
            self.particles.burst(portal.pos.x, portal.pos.y, portal.color, count=30, speed=300, life=0.7, size=6)
            self.particles.ring(portal.pos.x, portal.pos.y, portal.color, count=24, speed=200, life=0.5)
            self.sound.play("teleport"); self._flash(portal.color, 0.2)
            self.player.pos = pygame.math.Vector2(portal.dest_x, portal.dest_y)
            self.player.rect.center = (int(portal.dest_x), int(portal.dest_y))
            self.player.teleport_cooldown = 0.3; self.player.trail.clear()
            self.particles.burst(portal.dest_x, portal.dest_y, portal.color, count=20, speed=200, life=0.5, size=4)
            self.current_step += 1
            if self.current_step >= len(self.sequence):
                self.exit_gate.active = True; self.sound.play("level_up"); self._flash(EXIT_COLOR, 0.3)
            else:
                next_portal = self.portals[self.sequence[self.current_step]]
                next_portal.active = True
        else:
            self.sound.play("wrong"); self._shake(12, 0.5); self._flash((255, 50, 50), 0.3)
            self.particles.burst(self.player.pos.x, self.player.pos.y, (255, 50, 50), count=30, speed=350, life=0.6, size=5)
            self._reset_puzzle()

    def _on_exit_reached(self):
        self.state = LEVEL_COMPLETE; self.sound.play("exit")
        self.particles.burst(self.exit_gate.pos.x, self.exit_gate.pos.y, EXIT_COLOR, count=50, speed=400, life=1.0, size=7)
        self.particles.ring(self.exit_gate.pos.x, self.exit_gate.pos.y, WHITE, count=40, speed=250, life=0.8)
        self._flash(EXIT_COLOR, 0.5)

    def _draw(self):
        sx, sy = 0, 0
        if self.shake_intensity > 0 and self.shake_timer > 0:
            sx = random.randint(-self.shake_intensity, self.shake_intensity)
            sy = random.randint(-self.shake_intensity, self.shake_intensity)

        game_surf = pygame.Surface((WIDTH, HEIGHT))
        game_surf.fill(BLACK)
        for y in range(GRID_ROWS):
            for x in range(GRID_COLS):
                rx = x * TILE_SIZE; ry = y * TILE_SIZE + Y_OFFSET
                pygame.draw.rect(game_surf, FLOOR_COLOR, (rx, ry, TILE_SIZE, TILE_SIZE))
                pygame.draw.rect(game_surf, FLOOR_GRID, (rx, ry, TILE_SIZE, TILE_SIZE), 1)

        for w in self.walls:
            pygame.draw.rect(game_surf, WALL_FILL, w)
            pygame.draw.rect(game_surf, WALL_TOP, (w.x, w.y, w.w, 4))
            pygame.draw.rect(game_surf, WALL_EDGE, (w.x, w.y, 4, w.h))
            pygame.draw.rect(game_surf, WALL_EDGE, w, 1)

        if self.exit_gate: self.exit_gate.draw(game_surf, pygame.time.get_ticks())
        for p in self.portals: p.draw(game_surf, pygame.time.get_ticks())
        if self.player: self.player.draw(game_surf)
        self.particles.draw(game_surf)

        if self.hint_active and self.current_step < len(self.sequence):
            target = self.portals[self.sequence[self.current_step]]
            dx = target.pos.x - self.player.pos.x; dy = target.pos.y - self.player.pos.y
            dist = math.hypot(dx, dy)
            if dist > 1:
                ux, uy = dx / dist, dy / dist
                start = (self.player.pos.x + ux * 25, self.player.pos.y + uy * 25)
                end   = (self.player.pos.x + ux * 60, self.player.pos.y + uy * 60)
                pygame.draw.line(game_surf, target.color, start, end, 3)
                ang = math.atan2(uy, ux)
                for off in (-0.5, 0.5):
                    hx = end[0] - math.cos(ang + off) * 12; hy = end[1] - math.sin(ang + off) * 12
                    pygame.draw.line(game_surf, target.color, end, (hx, hy), 3)

        if self.flash_timer > 0 and self.flash_color:
            alpha = int(120 * (self.flash_timer / 0.5))
            flash = pygame.Surface((WIDTH, HEIGHT)); flash.set_alpha(max(0, min(255, alpha)))
            flash.fill(self.flash_color); game_surf.blit(flash, (0, 0))

        self.screen.fill(BLACK)
        self.screen.blit(game_surf, (sx, sy))
        self._draw_ui()

        if self.state == START_SCREEN: self._draw_start_screen()
        elif self.state == PAUSED: self._draw_pause()
        elif self.state == LEVEL_COMPLETE: self._draw_level_complete()
        elif self.state == GAME_COMPLETE: self._draw_game_complete()

        pygame.display.flip()

    def _draw_ui(self):
        pygame.draw.rect(self.screen, UI_BG, (0, 0, WIDTH, UI_HEIGHT))
        pygame.draw.line(self.screen, ACCENT, (0, UI_HEIGHT), (WIDTH, UI_HEIGHT), 2)
        lvl_text = self.font_md.render(f"LEVEL {self.level}", True, WHITE)
        self.screen.blit(lvl_text, (15, 18))
        diff_stars = "*" * self.difficulty
        diff_text = self.font_sm.render(f"Difficulty: {diff_stars}", True, ACCENT)
        self.screen.blit(diff_text, (15, 42))

        mins = int(self.level_time) // 60; secs = int(self.level_time) % 60
        time_text = self.font_md.render(f"{mins:02d}:{secs:02d}", True, WHITE)
        self.screen.blit(time_text, (WIDTH // 2 - time_text.get_width() // 2, 18))
        snd_text = self.font_sm.render("[M] Sound: " + ("ON" if self.sound.enabled else "OFF"), True, GRAY)
        self.screen.blit(snd_text, (WIDTH // 2 - snd_text.get_width() // 2, 42))

        seq_x = WIDTH - 20; seq_y = 30
        for i in range(len(self.sequence) - 1, -1, -1):
            portal_idx = self.sequence[i]; color = self.portals[portal_idx].color
            if i < self.current_step:
                pygame.draw.circle(self.screen, (60, 100, 60), (seq_x, seq_y), 10)
                pygame.draw.circle(self.screen, (100, 160, 100), (seq_x, seq_y), 10, 1)
            elif i == self.current_step:
                pygame.draw.circle(self.screen, color, (seq_x, seq_y), 12)
                pygame.draw.circle(self.screen, WHITE, (seq_x, seq_y), 12, 2)
            else:
                dim = tuple(c // 3 for c in color)
                pygame.draw.circle(self.screen, dim, (seq_x, seq_y), 9, 2)
            seq_x -= 28
        seq_label = self.font_xs.render("SEQUENCE →", True, GRAY)
        self.screen.blit(seq_label, (seq_x - seq_label.get_width() - 8, seq_y - 6))

        ctrl = self.font_xs.render("[WASD] Move  [H] Hint  [P] Pause  [R] Restart  [M] Mute  [ESC] Quit", True, (70, 70, 90))
        self.screen.blit(ctrl, (15, UI_HEIGHT - 16))

    def _draw_start_screen(self):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA); overlay.fill((0, 0, 0, 210)); self.screen.blit(overlay, (0, 0))
        title = self.font_lg.render("PORTAL SEQUENCE", True, ACCENT)
        title2 = self.font_lg.render("PUZZLE", True, WHITE)
        self.screen.blit(title, (WIDTH//2 - title.get_width()//2, 120))
        self.screen.blit(title2, (WIDTH//2 - title2.get_width()//2, 180))
        lines = [
            "Enter portals in the correct sequence to solve each level.",
            "Touching a WRONG portal resets you to the start.",
            "Complete all portals, then reach the green exit gate.",
            "",
            "Levels are validated via Monte Carlo Tree Search (MCTS).",
            "",
            "[WASD / Arrows] Move    [H] Hint (penalty)",
            "[P] Pause    [R] Restart    [M] Mute    [ESC] Quit",
        ]
        for i, line in enumerate(lines):
            color = WHITE if not line.startswith("  •") else (160, 160, 200)
            text = self.font_sm.render(line, True, color)
            self.screen.blit(text, (WIDTH//2 - text.get_width()//2, 260 + i * 24))
        pulse = abs(math.sin(pygame.time.get_ticks() * 0.003))
        press_color = (int(50 + 205*pulse), int(255), int(100 + 155*pulse))
        press = self.font_md.render("Press SPACE to Start", True, press_color)
        self.screen.blit(press, (WIDTH//2 - press.get_width()//2, HEIGHT - 80))

    def _draw_pause(self):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA); overlay.fill((0, 0, 0, 180)); self.screen.blit(overlay, (0, 0))
        text = self.font_lg.render("PAUSED", True, WHITE)
        self.screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT//2 - 40))
        sub = self.font_sm.render("Press P or ESC to resume", True, GRAY)
        self.screen.blit(sub, (WIDTH//2 - sub.get_width()//2, HEIGHT//2 + 20))

    def _draw_level_complete(self):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA); overlay.fill((0, 30, 0, 180)); self.screen.blit(overlay, (0, 0))
        text = self.font_lg.render("LEVEL COMPLETE!", True, EXIT_COLOR)
        self.screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT//2 - 80))
        mins = int(self.level_time) // 60; secs = int(self.level_time) % 60
        time_text = self.font_md.render(f"Time: {mins:02d}:{secs:02d}", True, WHITE)
        self.screen.blit(time_text, (WIDTH//2 - time_text.get_width()//2, HEIGHT//2 - 20))
        if self.level >= 10: next_text = self.font_md.render("Press SPACE for Final Results", True, ACCENT)
        else: next_text = self.font_md.render(f"Press SPACE for Level {self.level + 1}", True, ACCENT)
        self.screen.blit(next_text, (WIDTH//2 - next_text.get_width()//2, HEIGHT//2 + 40))

    def _draw_game_complete(self):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA); overlay.fill((20, 0, 30, 200)); self.screen.blit(overlay, (0, 0))
        text = self.font_lg.render("GAME COMPLETE!", True, ACCENT)
        self.screen.blit(text, (WIDTH//2 - text.get_width()//2, 150))
        mins = int(self.total_time) // 60; secs = int(self.total_time) % 60
        time_text = self.font_md.render(f"Total Time: {mins:02d}:{secs:02d}", True, WHITE)
        self.screen.blit(time_text, (WIDTH//2 - time_text.get_width()//2, 250))
        lvl_text = self.font_md.render(f"Levels Completed: {self.level}", True, EXIT_COLOR)
        self.screen.blit(lvl_text, (WIDTH//2 - lvl_text.get_width()//2, 300))
        restart = self.font_md.render("Press R or SPACE to Play Again", True, WHITE)
        self.screen.blit(restart, (WIDTH//2 - restart.get_width()//2, 420))

    def run(self):
        running = True
        while running:
            dt = min(self.clock.tick(FPS) / 1000.0, 1/30)
            for event in pygame.event.get():
                if not self._handle_event(event): running = False
            self._update(dt)
            self._draw()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    Game().run()