import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, List, Dict, Optional
import time
from matplotlib.animation import FuncAnimation

# Für Reproduzierbarkeit
np.random.seed(42)
torch.manual_seed(42)

# Globale Einstellungen
GRID_SIZE: int = 100
NUM_ROBOTS_PER_TEAM: int = 250  # Mindestens 500 Datenpunkte insgesamt
CENTER_ZONE: Tuple[int, int] = (GRID_SIZE // 2, GRID_SIZE // 2)
CENTER_ZONE_SIZE: int = 3  # Größe des Zentrums (3x3)

# Neue Parameter für erweiterte Funktionalität
MAX_STEPS: int = 5000  # Reduziert von 20000 auf realistischere Anzahl
SAVE_ANIMATION: bool = True  # Animation als GIF speichern
LEARNING_RATE: float = 0.001  # Lernrate für die Modelle
STRATEGY_UPDATE_INTERVAL: int = 100  # Wie oft die Strategie aktualisiert wird

class Environment:
    """
    Repräsentiert das Arena-Umfeld mit Hindernissen.
    """
    def __init__(self, size: int):
        self.size: int = size
        self.grid: np.ndarray = np.zeros((size, size))
        self.obstacle_positions: List[Tuple[int, int]] = []
        self.generate_obstacles()
        self.team_scores: Dict[str, int] = {"blue": 0, "red": 0}
        # Ressourcen hinzufügen
        self.resources: Dict[Tuple[int, int], int] = {}
        self.generate_resources()

    def generate_obstacles(self) -> None:
        """
        Platziert zufällig 200 Hindernisse auf dem Grid, wobei der CENTER_ZONE frei bleibt.
        """
        count: int = 0
        while count < 200:
            x, y = np.random.randint(0, self.size), np.random.randint(0, self.size)
            # Vermeide Hindernisse in der CENTER_ZONE
            if abs(x - CENTER_ZONE[0]) <= CENTER_ZONE_SIZE//2 and abs(y - CENTER_ZONE[1]) <= CENTER_ZONE_SIZE//2:
                continue
            if self.grid[x, y] == -1:
                continue
            self.grid[x, y] = -1  # -1 steht für ein Hindernis
            self.obstacle_positions.append((x, y))
            count += 1
            
    def generate_resources(self) -> None:
        """
        Platziert zufällig 50 Ressourcen auf dem Grid, nicht auf Hindernissen oder im CENTER_ZONE.
        """
        count: int = 0
        while count < 50:
            x, y = np.random.randint(0, self.size), np.random.randint(0, self.size)
            # Vermeide Ressourcen auf Hindernissen oder im CENTER_ZONE
            if self.grid[x, y] == -1 or (abs(x - CENTER_ZONE[0]) <= CENTER_ZONE_SIZE//2 and 
                                         abs(y - CENTER_ZONE[1]) <= CENTER_ZONE_SIZE//2):
                continue
            self.resources[(x, y)] = np.random.randint(10, 30)  # Ressourcenwert zwischen 10-30
            self.grid[x, y] = 0.5  # Ressourcen haben Wert 0.5 im Grid
            count += 1

    def get_state(self) -> torch.Tensor:
        """
        Gibt den aktuellen Zustand des Grids als Tensor zurück.
        """
        return torch.tensor(self.grid, dtype=torch.float32)
    
    def update_score(self, team: str, points: int) -> None:
        """
        Aktualisiert die Punktzahl eines Teams.
        """
        self.team_scores[team] += points

class DiffusionModel(nn.Module):
    """
    Ein verbessertes Feedforward-Netz zur Generierung neuer Umweltvorhersagen aus Schwarmdaten.
    """
    def __init__(self, input_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(0.2)  # Dropout zur Vermeidung von Overfitting
        self.fc2 = nn.Linear(128, 256)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, input_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class Robot:
    """
    Repräsentiert einen einzelnen Roboter in der Arena.
    """
    def __init__(self, env: Environment, team: str, id: int):
        self.env: Environment = env
        self.team: str = team
        self.id: int = id  # Eindeutige ID für jeden Roboter
        self.position: np.ndarray = self.initialize_position()
        self.energy: float = 100.0  # Startenergie
        self.collected_resources: int = 0
        self.memory: List[Tuple[np.ndarray, float]] = []  # Speichert vergangene Positionen und Messungen
        self.memory_size: int = 10  # Größe des Gedächtnisses
        self.strategy: str = "explore"  # Standardstrategie: erkunden
    
    def initialize_position(self) -> np.ndarray:
        """
        Initialisiert eine Position, die nicht auf einem Hindernis liegt und
        je nach Team auf einer bestimmten Seite des Grids.
        """
        while True:
            if self.team == "blue":
                x = np.random.randint(0, GRID_SIZE // 3)  # Blaue starten links
            else:
                x = np.random.randint(2 * GRID_SIZE // 3, GRID_SIZE)  # Rote starten rechts
            y = np.random.randint(0, GRID_SIZE)
            if self.env.grid[x, y] != -1:
                return np.array([x, y])

    def move(self, swarm_positions: Optional[List[np.ndarray]] = None) -> None:
        """
        Bewegt den Roboter basierend auf seiner aktuellen Strategie.
        """
        # Energie beim Bewegen reduzieren
        self.energy -= 0.1
        
        # Bei zu niedriger Energie zurück ins Zentrum
        if self.energy < 20:
            self.strategy = "return"
            
        # Bewegungslogik je nach Strategie
        if self.strategy == "explore":
            self._move_explore()
        elif self.strategy == "collect":
            self._move_collect()
        elif self.strategy == "return":
            self._move_return()
        elif self.strategy == "swarm":
            self._move_swarm(swarm_positions)
            
        # Überprüfe Ressourcen an aktueller Position
        self._check_for_resources()
        
        # Füge aktuelle Position und Messung zum Gedächtnis hinzu
        measurement = self.sense_environment()
        self.memory.append((self.position.copy(), measurement))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)  # Entferne ältesten Eintrag wenn Gedächtnis voll

    def _move_explore(self) -> None:
        """
        Zufällige Erkundungsbewegung mit leichter Präferenz für unerforschte Gebiete.
        """
        # Basis-Bewegungsvektor (zufällig)
        move_vector = np.random.choice([-1, 0, 1], size=2)
        
        # Vermeide bereits besuchte Positionen wenn möglich
        visited_positions = [mem[0] for mem in self.memory]
        attempts = 0
        while any(np.array_equal(self.position + move_vector, pos) for pos in visited_positions) and attempts < 5:
            move_vector = np.random.choice([-1, 0, 1], size=2)
            attempts += 1
            
        self._apply_move(move_vector)

    def _move_collect(self) -> None:
        """
        Bewegt sich in Richtung der nächsten bekannten Ressource.
        """
        # Suche nach der nächsten Ressource
        min_dist = float('inf')
        target = None
        
        for pos, value in self.env.resources.items():
            if value > 0:  # Nur wenn Ressource noch vorhanden
                x, y = pos
                dist = np.linalg.norm(self.position - np.array([x, y]))
                if dist < min_dist:
                    min_dist = dist
                    target = np.array([x, y])
        
        if target is not None:
            # Bewege dich in Richtung der Ressource
            direction = np.clip(target - self.position, -1, 1)
            direction = np.round(direction).astype(int)  # Rundung auf -1, 0, 1
            self._apply_move(direction)
        else:
            # Keine Ressource gefunden, zurück zur Erkundung
            self.strategy = "explore"
            self._move_explore()

    def _move_return(self) -> None:
        """
        Bewegt sich in Richtung CENTER_ZONE, um Energie aufzuladen oder Ressourcen abzuliefern.
        """
        # Richtung zum Zentrum
        center = np.array(CENTER_ZONE)
        direction = np.clip(center - self.position, -1, 1)
        direction = np.round(direction).astype(int)  # Rundung auf -1, 0, 1
        self._apply_move(direction)
        
        # Wenn im CENTER_ZONE angekommen
        if abs(self.position[0] - CENTER_ZONE[0]) <= CENTER_ZONE_SIZE//2 and \
           abs(self.position[1] - CENTER_ZONE[1]) <= CENTER_ZONE_SIZE//2:
            # Energie aufladen
            self.energy = min(100, self.energy + 10)
            
            # Ressourcen abliefern
            if self.collected_resources > 0:
                self.env.update_score(self.team, self.collected_resources)
                self.collected_resources = 0
                
            # Zurück zur Erkundung wenn Energie ausreichend
            if self.energy > 80:
                self.strategy = "explore"

    def _move_swarm(self, swarm_positions: List[np.ndarray]) -> None:
        """
        Bewegt sich in Richtung des Schwerpunkts der anderen Roboter des Teams.
        """
        if swarm_positions and len(swarm_positions) > 0:
            # Berechne Schwerpunkt, ohne die eigene Position
            other_positions = [pos for pos in swarm_positions if not np.array_equal(pos, self.position)]
            if other_positions:
                centroid = np.mean(other_positions, axis=0)
                direction = np.clip(centroid - self.position, -1, 1)
                direction = np.round(direction).astype(int)
                self._apply_move(direction)
            else:
                self._move_explore()
        else:
            self._move_explore()

    def _apply_move(self, move_vector: np.ndarray) -> None:
        """
        Wendet einen Bewegungsvektor an und stellt sicher, dass der Roboter auf dem Grid bleibt
        und nicht in Hindernisse läuft.
        """
        new_position = self.position + move_vector
        new_position = np.clip(new_position, 0, GRID_SIZE - 1)
        
        # Überprüfe, ob die neue Position frei ist
        x, y = new_position.astype(int)
        if self.env.grid[x, y] != -1:  # Kein Hindernis
            self.position = new_position

    def _check_for_resources(self) -> None:
        """
        Überprüft, ob an der aktuellen Position Ressourcen sind und sammelt sie ggf. ein.
        """
        pos_tuple = (int(self.position[0]), int(self.position[1]))
        if pos_tuple in self.env.resources and self.env.resources[pos_tuple] > 0:
            # Ressource einsammeln
            resource_value = self.env.resources[pos_tuple]
            self.collected_resources += resource_value
            self.env.resources[pos_tuple] = 0  # Ressource verbraucht
            self.env.grid[pos_tuple] = 0  # Grid aktualisieren
            
            # Wechsle zur Rückkehrstrategie, wenn viele Ressourcen gesammelt wurden
            if self.collected_resources > 50:
                self.strategy = "return"

    def sense_environment(self) -> float:
        """
        Misst den Wert der aktuellen Zelle.
        """
        x, y = self.position.astype(int)
        return self.env.grid[x, y]
    
    def update_strategy(self, nearby_robots: List['Robot']) -> None:
        """
        Aktualisiert die Strategie basierend auf Umgebungsfaktoren und benachbarten Robotern.
        """
        # Zähle Roboter des eigenen und gegnerischen Teams in der Nähe
        own_team_count = 0
        enemy_team_count = 0
        
        for robot in nearby_robots:
            if robot.team == self.team:
                own_team_count += 1
            else:
                enemy_team_count += 1
        
        # Strategie-Entscheidungslogik
        if self.energy < 20:
            # Bei niedriger Energie zurück zum Aufladen
            self.strategy = "return"
        elif self.collected_resources > 50:
            # Bei vielen Ressourcen zurück zum Abliefern
            self.strategy = "return"
        elif enemy_team_count > own_team_count + 2:
            # Bei Feindübermacht: Schwarm-Verhalten
            self.strategy = "swarm"
        elif any(pos_tuple in self.env.resources and self.env.resources[pos_tuple] > 0 
                for pos_tuple in [(int(self.position[0]+dx), int(self.position[1]+dy)) 
                                 for dx in [-2,-1,0,1,2] for dy in [-2,-1,0,1,2]]):
            # Ressourcen in der Nähe: Sammeln
            self.strategy = "collect"
        else:
            # Standardfall: Erkunden
            self.strategy = "explore"

def swarm_collect_data(robots: List[Robot]) -> torch.Tensor:
    """
    Sammelt Schwarmdaten von einer Liste von Robotern.
    Jeder Datenpunkt besteht aus (x, y, gemessener Zustand).
    """
    data = [(robot.position[0], robot.position[1], robot.sense_environment()) for robot in robots]
    return torch.tensor(data, dtype=torch.float32)

def find_nearby_robots(robot: Robot, all_robots: List[Robot], radius: int = 5) -> List[Robot]:
    """
    Findet alle Roboter in einem bestimmten Radius um den gegebenen Roboter.
    """
    nearby = []
    for other in all_robots:
        if other.id != robot.id:  # Ignoriere den Roboter selbst
            distance = np.linalg.norm(robot.position - other.position)
            if distance <= radius:
                nearby.append(other)
    return nearby

def train_diffusion_model(model: DiffusionModel, data: torch.Tensor) -> None:
    """
    Trainiert das Diffusionsmodell mit den gesammelten Daten.
    """
    # Einfaches unüberwachtes Lernen: Versuche, die Eingabedaten zu reproduzieren
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # Einige Trainingsschritte
    for _ in range(5):  # Begrenzter Trainingszyklus pro Spielschritt
        optimizer.zero_grad()
        output = model(data)
        # Hier könnten komplexere Verlustfunktionen implementiert werden
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()

def generate_new_states(diff_model: DiffusionModel, data: torch.Tensor) -> torch.Tensor:
    """
    Nutzt das Diffusionsmodell, um aus den Schwarmdaten neue Umweltvorhersagen zu generieren.
    """
    with torch.no_grad():
        return diff_model(data)

def plot_environment(env: Environment, robots_blue: List[Robot], robots_red: List[Robot], 
                    step: int, ax: plt.Axes) -> None:
    """
    Visualisiert das aktuelle Grid, Hindernisse, den CENTER_ZONE und die Positionen der Roboter.
    """
    ax.clear()
    # Hintergrundbild: Das Grid als Bild anzeigen
    ax.imshow(env.grid.T, origin='lower', cmap='gray_r', extent=[0, GRID_SIZE, 0, GRID_SIZE])
    
    # Explizite Darstellung der Hindernisse
    for x, y in env.obstacle_positions:
        rect = patches.Rectangle((x, y), 1, 1, linewidth=0, facecolor='black')
        ax.add_patch(rect)
    
    # Ressourcen anzeigen
    for (x, y), value in env.resources.items():
        if value > 0:  # Nur nicht verbrauchte Ressourcen anzeigen
            # Größe und Farbe je nach Wert
            size = max(5, min(20, value))
            alpha = max(0.3, min(0.9, value / 30))
            ax.scatter(x + 0.5, y + 0.5, color='green', s=size, alpha=alpha)
    
    # Zeichne den CENTER_ZONE
    cx, cy = CENTER_ZONE
    cz_size = CENTER_ZONE_SIZE
    center_rect = patches.Rectangle((cx - cz_size//2, cy - cz_size//2), cz_size, cz_size, 
                                   linewidth=2, edgecolor='gold', facecolor='yellow', alpha=0.5)
    ax.add_patch(center_rect)
    
    # Zeichne Roboter: blaues Team mit Energieanzeige
    for robot in robots_blue:
        # Größe je nach Energie
        size = max(5, min(15, robot.energy / 10))
        alpha = max(0.3, min(0.9, robot.energy / 100))
        ax.scatter(robot.position[0] + 0.5, robot.position[1] + 0.5, 
                  color='blue', s=size, alpha=alpha)
    
    # Zeichne Roboter: rotes Team mit Energieanzeige
    for robot in robots_red:
        # Größe je nach Energie
        size = max(5, min(15, robot.energy / 10))
        alpha = max(0.3, min(0.9, robot.energy / 100))
        ax.scatter(robot.position[0] + 0.5, robot.position[1] + 0.5, 
                  color='red', s=size, alpha=alpha)

    # Anzeige der aktuellen Punktestände
    score_text = f"Blau: {env.team_scores['blue']} - Rot: {env.team_scores['red']}"
    ax.text(5, GRID_SIZE - 5, score_text, fontsize=12, color='black', 
            bbox=dict(facecolor='white', alpha=0.7))

    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_title(f"KI vs. KI Schwarm-Kampf – Schritt {step}")
    ax.grid(True)

def run_simulation() -> None:
    """
    Startet die Simulation: Zwei Teams von Robotern bewegen sich in der Arena,
    während Diffusionsmodelle experimentell neue Umweltvorhersagen generieren.
    """
    env = Environment(GRID_SIZE)
    
    # Roboter mit eindeutigen IDs erstellen
    robots_blue = [Robot(env, "blue", i) for i in range(NUM_ROBOTS_PER_TEAM)]
    robots_red = [Robot(env, "red", i + NUM_ROBOTS_PER_TEAM) for i in range(NUM_ROBOTS_PER_TEAM)]
    
    diff_model_blue = DiffusionModel(3)
    diff_model_red = DiffusionModel(3)

    # Plot initialisieren
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Für Animation
    frames = []
    
    start_time = time.time()
    
    for step in range(MAX_STEPS):
        # Berechne Schwarm-Positionen für jedes Team
        blue_positions = [robot.position for robot in robots_blue]
        red_positions = [robot.position for robot in robots_red]
        
        # Strategie-Updates alle STRATEGY_UPDATE_INTERVAL Schritte
        if step % STRATEGY_UPDATE_INTERVAL == 0:
            all_robots = robots_blue + robots_red
            for robot in all_robots:
                nearby = find_nearby_robots(robot, all_robots)
                robot.update_strategy(nearby)
        
        # Bewegung beider Teams
        for robot in robots_blue:
            robot.move(blue_positions)
        for robot in robots_red:
            robot.move(red_positions)
        
        # Schwarmdaten sammeln
        swarm_data_blue = swarm_collect_data(robots_blue)
        swarm_data_red = swarm_collect_data(robots_red)
        
        # Diffusionsmodelle trainieren und neue Zustände generieren
        train_diffusion_model(diff_model_blue, swarm_data_blue)
        train_diffusion_model(diff_model_red, swarm_data_red)
        
        generated_states_blue = generate_new_states(diff_model_blue, swarm_data_blue)
        generated_states_red = generate_new_states(diff_model_red, swarm_data_red)
        
        # Update der Umwelt basierend auf den generierten Zuständen
        for x, y, val in generated_states_blue.numpy():
            xi, yi = int(np.clip(x, 0, GRID_SIZE - 1)), int(np.clip(y, 0, GRID_SIZE - 1))
            if env.grid[xi, yi] != -1 and not (abs(xi - CENTER_ZONE[0]) <= CENTER_ZONE_SIZE//2 and 
                                             abs(yi - CENTER_ZONE[1]) <= CENTER_ZONE_SIZE//2):
                env.grid[xi, yi] = (env.grid[xi, yi] + val) / 2

        for x, y, val in generated_states_red.numpy():
            xi, yi = int(np.clip(x, 0, GRID_SIZE - 1)), int(np.clip(y, 0, GRID_SIZE - 1))
            if env.grid[xi, yi] != -1 and not (abs(xi - CENTER_ZONE[0]) <= CENTER_ZONE_SIZE//2 and 
                                             abs(yi - CENTER_ZONE[1]) <= CENTER_ZONE_SIZE//2):
                env.grid[xi, yi] = (env.grid[xi, yi] + val) / 2

        # Aktualisiere die Visualisierung
        plot_environment(env, robots_blue, robots_red, step, ax)
        
        # Für Animation speichern
        if SAVE_ANIMATION and step % 10 == 0:  # Speichere nur jeden 10. Frame
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            frames.append(frame)
        
        plt.pause(0.01)
        
        # Ausgabe des Fortschritts
        if step % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Schritt {step}/{MAX_STEPS} ({step/MAX_STEPS*100:.1f}%) - "
                  f"Zeit: {elapsed:.1f}s - Score: Blau {env.team_scores['blue']} : {env.team_scores['red']} Rot")
            
            # Ressourcen regelmäßig nachfüllen
            if step % 500 == 0 and step > 0:
                env.generate_resources()
                print("Neue Ressourcen generiert!")
    
    plt.ioff()
    
    # Zeige Endergebnis an
    winner = "Blau" if env.team_scores["blue"] > env.team_scores["red"] else "Rot"
    if env.team_scores["blue"] == env.team_scores["red"]:
        winner = "Unentschieden"
    
    print(f"\nSimulation beendet nach {MAX_STEPS} Schritten.")
    print(f"Endstand: Blau {env.team_scores['blue']} : {env.team_scores['red']} Rot")
    print(f"Sieger: {winner}")
    
    # Animation speichern
    if SAVE_ANIMATION and frames:
        from matplotlib.animation import ArtistAnimation
        print("Erstelle Animation...")
        animation = ArtistAnimation(fig, 
                                    [[plt.imshow(frame)] for frame in frames], 
                                    interval=100, 
                                    blit=True)
        animation.save('swarm_simulation.gif', writer='pillow', fps=10)
        print("Animation gespeichert als 'swarm_simulation.gif'")
    
    plt.show()

if __name__ == "__main__":
    run_simulation()
