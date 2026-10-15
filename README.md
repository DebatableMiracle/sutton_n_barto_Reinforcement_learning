# Reinforcement Learning: An introduction by Sutton & Barto

This is my repository for solving examples by code from the book Sutton and Barto 


## Racetrack problem (Example 5.12 Monte Carlo Methods)
![alt text](images/racetrack/problem_figure_55.png)

![alt text](images//racetrack/512exercise1.png)

![alt text](images/racetrack/512exercise2.png)

#### Solution: racetrack.ipynb

Firstly we build the racetrack from the book, the idea is to break down the track into 6 rectangles

Here's how to breakdown the first track into the 6 pieces

![alt text](images/racetrack/racetrack_breakdown.png)

you can use similar breakdown for the second track and use that in the notebook, just replace the rectangles

I only employ On Policy Monte Carlo (first visit) yet. Would do Off policy soon!

Here's some trajectories with noise = 0.1 injected! I was thinking of increasing noise to make the policy less "hug the wall" solution

![training](images/racetrack/training.png)
![training curves](images/racetrack/racetrack_training_curves.png)


## Windy Gridworld problem (Example 6.5 Temporal Difference Learning)

![alt text](images/windy_gridworld/example_65.png)
![alt text](images/windy_gridworld/exercise610.png)

Soo it's a relatively easier gridworld, I use 3 different action modes 4, 8, 9 which represent action_space size. 4 meaning classic up down left right, 8 includes diagonal actions, while 9 also includes a 0,0 action that is stay still.

![alt text](images/windy_gridworld/base_training_graph.png)

This graph shows a cool property of TD learning, it explores till around 2000 episodes while no results/termination. Monte Carlo would've failed in such a setting since it needs the final estimate of G, while this is bootstrap of return G. that is, Monte Carlo would require the episode to end to actually have a reward signal while here it uses estimates. 

Then there's policies learnt for different modes

### Actions mode = 4
approximate steps = 15
![alt text](images/windy_gridworld/trajectory_mode4.png)
### Actions mode = 8
approximate steps = 8
![alt text](images/windy_gridworld/trajectory_mode8.png)
### Actions mode = 9
approximate steps = 8
![alt text](images/windy_gridworld/trajectory_mode9.png)

## Cliffwalking Problem