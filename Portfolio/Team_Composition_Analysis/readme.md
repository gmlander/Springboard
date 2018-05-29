# Team Compositions in Professional League of Legends
***

The data being used was obtained from [Oracle's Elixer] (using upload from September 18th, 2017). A [codebook] for the variables is also included.

- **Part 1: Data Cleaning** -- can be found [here](http://nbviewer.jupyter.org/github/gmlander/Springboard/blob/master/Portfolio/Team_Composition_Analysis/I_Data_Cleaning.ipynb) or at the raw [github version](https://github.com/gmlander/Springboard/blob/master/Portfolio/Team_Composition_Analysis/I_Data_Cleaning.ipynb).
- **Part 2: Exploratory Data Analysis** -- can be found [here](http://nbviewer.jupyter.org/github/gmlander/Springboard/blob/master/Portfolio/Team_Composition_Analysis/II_Exploratory_Data_Analysis.ipynb) or at the [github version](https://github.com/gmlander/Springboard/blob/master/Portfolio/Team_Composition_Analysis/II_Exploratory_Data_Analysis.ipynb)
- **Part 3: Feature Engineering** -- can be found [here](http://nbviewer.jupyter.org/github/gmlander/Springboard/blob/master/Portfolio/Team_Composition_Analysis/III_Feature_Engineering.ipynb) or at the [github version](https://github.com/gmlander/Springboard/blob/master/Portfolio/Team_Composition_Analysis/III_Feature_Engineering.ipynb).
- **Part 4: Cluster Validation Testing** -- can be found [here](http://nbviewer.jupyter.org/github/gmlander/Springboard/blob/master/Portfolio/Team_Composition_Analysis/IV_Clustering_Validation_Tests.ipynb) or at the [github version](https://github.com/github/gmlander/Springboard/blob/master/Portfolio/Team_Composition_Analysis/IV_Clustering_Validation_Tests.ipynb)



[Oracle's Elixer]: http://oracleselixir.com/match-data/
[codebook]: http://oracleselixir.com/match-data/match-data-dictionary/

---


### Problem Definition and Intended Audience

This analysis evaluates game data from professional level _League of Legends_ (LoL) matches to find prredictors, patterns, and models that may aid a franchise in making pregame strategy decisions in regards to style of play and team composition, both generally and opponent-specifically. As the intended audience is a professional team, this analysis does not make much effort in explaining endemic terminology of the game. For a primer see [here](http://www.espn.com/esports/story/_/id/14545779/guide-league-legends) or [here](https://www.riftherald.com/2016/9/29/13027318/lol-guide-how-to-watch-play-intro).

The main impediment to analyzing League, or eSports in general, in the same manner as a traditional sport, is that it isn't one. Obviously. But it's an important thing to remember. Athletes can’t change their physical strengths and weaknesses from game to game, but League players can and do (or at least their champion's). So how does one predict the effectiveness of a team when its compositional elements are indefinite? That's the question this analysis hopes to answer, through machine learning techniques to bucketize and map the many different observed champion-playstyle combinations into a few discrete, interprettable archetypes that can be used as easily inform drafting/banning decisions.

---
