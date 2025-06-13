# Pràctica 3

> Autors: Pol Ribera Moreno, Judit Donate Castillo, Carlos Rosales Santos
---
## Contingut

1. `configs`: Configuracions dels paràmetres utilitzades en els experiments
2. `data`: Resultats dels experiments
3. `plots`: Gràfiques extretes a partir dels resultats
4. `reders`: Animacions dels experiments

## Execució

```bash
python main.py --config config.json --datadir dir
```
1. Càrrega una configuració des de l'arxiu `config.json`
2. Guarda els resultats de l'experiment en un arxiu `.json` en `data/dir/`. `dir` pot ser `minimax`, `nash`, `pareto` o `welfare`

Exemple:
```bash
python main.py --config configs/pareto/config.json --datadir pareto
```
---

## Paràmetres de configuració

Hem agrupat els paràmetres en les seguents categories

### 1. Configuració de l'entorn

- `size`: Mida del mapa, on el mapa és un quadrat de `size x size`
- `obstacle_density`: Densitat d'obstacles al mapa. Valor entre `0` y `1`, representa la probabilitat de que una cel·la sigui un obstacle.
- `maps`: Nombre de mapes diferents per entrenar i avaluar. Si el nombre total d'episodis excedeix aquest valor, els mapes es reutilitzen.
- `num_agents`: Nombre d'agents en l'entorn.
- `num_states`: Nombre total d'estats possibles. Es calcula com `obstacle_representation × agent_representation × target_representation`. Per exemple: `16 * 16 * 4`.
- `obs_radius`: Cada agent rep observacions corresponents a una quadrícula (x − r, y − r, x + r, y + r), on (x, y) és la posició de l'agent després de l'ultima acció i r és el radi d'observació.

---

### 2. Paràmetres d'Entrenament

- `epochs`: Nombre total d'èpoques d'entrenament. Cada època inclou diversos episodis i una avaluació.
- `episodes_per_epoch`: Nombre mínim d'episodis per època.
- `episode_length`: Longitud màxima de cada episodi, en passos. Si s'aconsegueix aquest límit, l'episodi es trunca.

---

### 3. Paràmetres de l'Algorisme

- `learning_rate`: Taxa d'aprenentatge (`alpha`). Controla quant s'actualitza el coneixement després de cada pas.
- `gamma`: Factor de descompte. Determina la importància de les recompenses futures.
- `epsilon_max`: Valor inicial de `epsilon` per època. Representa la probabilitat d'explorar al començament de l'entrenament.
- `epsilon_min`: Valor mínim de `epsilon`. Limita quant pot reduir-se l'exploració.
---

### 4. Visualització

- `save_every`: Freqüència (en nombre d'èpoques) amb la qual es guarda una animació SVG de l'entorn. Si és `None`, no es guarda automàticament.
- `renders`: Directori on es guarden les animacions de l'entorn en format SVG.

---

### 5. Concepte de Solució

- `solution_concept`: Classe que defineix el criteri de solució per a avaluar i comparar polítiques.
