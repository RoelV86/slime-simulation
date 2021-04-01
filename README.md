# Slime simulation
After watching [Sebastian Lague's video](https://www.youtube.com/watch?v=X-iSQQgOd1A&ab_channel=SebastianLague) about ant and slime simulation I wrote a similar implementation in Python. His video shows some really awesome results, especially the larger scale simulations (2560x1440 pixel map with 1M agents). The example below of this project is on a 400x400 map with 2500 agents.

Each agent moves and during movement it leaves a trail. An agent may turn if its sensors detect a trail, see [Sage Jenson's page](https://sagejenson.com/physarum) for a nice diagram of steps of a simulation tick and more cool results.

![Slime simulation](output.gif)

## References
- https://www.youtube.com/watch?v=X-iSQQgOd1A&ab_channel=SebastianLague
- https://sagejenson.com/physarum

