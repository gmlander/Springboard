# Capstone Proposal - Analysis of Professional *League of Legends* Matches

I would like to evaluate game data from professional level _League of Legends_ (LoL) matches with the intention of unearthing strategic insights that would benefit a professional LoL eSports franchise. I would like to break this analysis into two separate evaluations - _Pre-Game Strategic Adjustments_ and _Roster Composition Decisions_.

+ **Pre-Game Strategic Adjustments**

I would like to investigate if there are playstyle strategies available to teams that will improve their likelihood of winning. In american football for instance, bad teams - meaning those with lower records - are statistically more likely to beat a good opponent by raising the variance of the game through things like trick plays, onside kicks, aggressive throw selection, 4th down calls, etc. I'm interested in seeing if there are extensions of this in competitive League of Legends.

For instance, are the best teams more or less likely to win longer than average games? What about 'bloodier' (higher than average combined kills) games. To answer those questions, I will compare the win rate of top quantiles of teams (ordered by win percentage) in games _long_ games and _bloody_ games against their aggregate win rate to determine if there is a statistically significant difference in their performances.

Additionally, I may evaluate the win rate among champion sets, meaning, when x, y, and z champion or on the same team, do they have a higher than average likelihood of winning? I am somewhat hesitent to implement this last analysis because the competitive advantage of individual champions (and by extension champion sets) can change drastically with each game patch iteration. That means the breadth of my data would not only become much less robust were I to filter it by the most recent patch, but I would also run the risk of my isights quickly going stale. After all, if I had a model to predict tomorrow's stock price for a now defunct company, you would not be particularly interested in it.

+ **Roster Composition Decisions**

Just as there are predictors for a collective team's chance of winning, many of those same predictors can be applied to individual players. And just as teams are looking for advantages in the strategies they introduce in each game, they are also looking for advantages in the roster decisions they make - _"Does the player I currently have at this position give my team the best chance of winning?"_

Unfortunately, this analysis becomes quite difficult in a dynamic team sport. _Moneyball_'ing started in baseball before other professional sports for a multitude of reasons, but one of the main reasons was the nature of the game. Individual player performances of baseball teammates are much less correlated than in dynamic sports like basketball. For instance, a player's chance of getting a hit is not strongly correlated with how many hits his teammates have gotten (there is of course some weak correlation owed to the quality of the pitching they've been facing and the conditions at the park). The same cannot be said of dynamic sports though. If a basketball player misses a shot, there are a multitude of factors that influenced it - Did she take a bad shot because her teammates failed to set proper screens? Was it late in the shot-clock because her teammates couldn't move the ball properly for a better shot? Was she double teamed because one of her teammates has been missing shots all night? Etc.

_LoL_ player performance among teammates has similar analysis hurdles. But unlike basketball, _LoL_ players spend the early portion of their games playing mostly in isolation against their lane opponent(s). This gives a window for evaluating individual player performance effect on win rate before the dynamic team nature of the game muffles the signal. Therefore I will focus my analysis on early-game statistics of individual players as a determinant for talent assesment and roster composition decisions.

### Data Source and Acquisition

The data will be obtained from [Oracle's Elixer]. It is composed of several wide (92 colums) xlsx files that can easily be read and concatenated into a pandas dataframe. The total observations in the dataset are approximately 45k. A [codebook] for the variables is also included.
[Oracle's Elixer]: http://oracleselixir.com/match-data/
[codebook]: http://oracleselixir.com/match-data/match-data-dictionary/


### Deliverables

